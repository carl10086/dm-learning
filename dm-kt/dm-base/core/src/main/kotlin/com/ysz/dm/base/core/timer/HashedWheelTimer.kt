package com.ysz.dm.base.core.timer

import org.slf4j.LoggerFactory
import java.util.*
import java.util.concurrent.*
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicIntegerFieldUpdater
import java.util.concurrent.atomic.AtomicLong

/**
 * 从 netty 那里 copy 来的 hash 环定时器: 本质上来源于操作系统内核的调度算法 .
 *
 * 算法 paper: http://www.cs.columbia.edu/~nahum/w6998/papers/sosp87-timing-wheels.pdf
 *
 *
 *
 *  * 把队列从 jctools 的改成了 ConcurrentLinkedQueue
 *  * 实例个数最多允许 4个目前
 *
 *
 */
class HashedWheelTimer @JvmOverloads constructor(
//    threadFactory: ThreadFactory? = Executors.defaultThreadFactory(),
    threadFactory: ThreadFactory = object : ThreadFactory {
        private val threadNumber = AtomicInteger(1)
        override fun newThread(r: Runnable): Thread {
            return Thread(r).apply {
                this.name = "HashedWheelTimer:${threadNumber.getAndIncrement()}"
                this.isDaemon = true
            }
        }
    },
    tickDuration: Long = 100,
    unit: TimeUnit? = TimeUnit.MILLISECONDS,
    ticksPerWheel: Int = 512,
    maxPendingTimeouts: Long = -1
) : Timer {
    private val worker: Worker = Worker()
    private val workerThread: Thread

    @Suppress("unused")
    @Volatile
    private var workerState = 0 // 0 - init, 1 - started, 2 - shut down
    private val tickDuration: Long
    private val wheel: Array<HashedWheelBucket?>
    private val mask: Int
    private val startTimeInitialized = CountDownLatch(1)

    /**
     * 这里跟 netty 不同、netty 的队列是 jctools 里的高性能队列、暂时不想引入那个库、看 netty 的使用姿势、那个库某些场景下有店问题，需要根据操作系统来写 if else 优化
     */
    private val timeouts: Queue<HashedWheelTimeout> = ConcurrentLinkedQueue()
    private val cancelledTimeouts: Queue<HashedWheelTimeout> = ConcurrentLinkedQueue()
    private val pendingTimeouts = AtomicLong(0)
    private val maxPendingTimeouts: Long

    @Volatile
    private var startTime: Long = 0

    /**
     * Creates a new timer with the default thread factory
     * ([Executors.defaultThreadFactory]) and default number of ticks
     * per wheel.
     *
     * @param tickDuration the duration between tick
     * @param unit         the time unit of the `tickDuration`
     * @throws NullPointerException     if `unit` is `null`
     * @throws IllegalArgumentException if `tickDuration` is &lt;= 0
     */
    constructor(tickDuration: Long, unit: TimeUnit?) : this(Executors.defaultThreadFactory(), tickDuration, unit)

    /**
     * Creates a new timer with the default thread factory
     * ([Executors.defaultThreadFactory]).
     *
     * @param tickDuration  the duration between tick
     * @param unit          the time unit of the `tickDuration`
     * @param ticksPerWheel the size of the wheel
     * @throws NullPointerException     if `unit` is `null`
     * @throws IllegalArgumentException if either of `tickDuration` and `ticksPerWheel` is &lt;= 0
     */
    constructor(tickDuration: Long, unit: TimeUnit?, ticksPerWheel: Int) : this(
        Executors.defaultThreadFactory(),
        tickDuration,
        unit,
        ticksPerWheel
    )
    /**
     * Creates a new timer.
     *
     * @param threadFactory      a [ThreadFactory] that creates a
     * background [Thread] which is dedicated to
     * [TimerTask] execution.
     * @param tickDuration       the duration between tick
     * @param unit               the time unit of the `tickDuration`
     * @param ticksPerWheel      the size of the wheel
     * @param maxPendingTimeouts The maximum number of pending timeouts after which call to
     * `newTimeout` will result in
     * [RejectedExecutionException]
     * being thrown. No maximum pending timeouts limit is assumed if
     * this value is 0 or negative.
     * @throws NullPointerException     if either of `threadFactory` and `unit` is `null`
     * @throws IllegalArgumentException if either of `tickDuration` and `ticksPerWheel` is &lt;= 0
     */
    /**
     * Creates a new timer.
     *
     * @param threadFactory a [ThreadFactory] that creates a
     * background [Thread] which is dedicated to
     * [TimerTask] execution.
     * @param tickDuration  the duration between tick
     * @param unit          the time unit of the `tickDuration`
     * @param ticksPerWheel the size of the wheel
     * @throws NullPointerException     if either of `threadFactory` and `unit` is `null`
     * @throws IllegalArgumentException if either of `tickDuration` and `ticksPerWheel` is &lt;= 0
     */
    /**
     * Creates a new timer with the default number of ticks per wheel.
     *
     * @param threadFactory a [ThreadFactory] that creates a
     * background [Thread] which is dedicated to
     * [TimerTask] execution.
     * @param tickDuration  the duration between tick
     * @param unit          the time unit of the `tickDuration`
     * @throws NullPointerException     if either of `threadFactory` and `unit` is `null`
     * @throws IllegalArgumentException if `tickDuration` is &lt;= 0
     */
    /**
     * Creates a new timer with the default tick duration and default number of
     * ticks per wheel.
     *
     * @param threadFactory a [ThreadFactory] that creates a
     * background [Thread] which is dedicated to
     * [TimerTask] execution.
     * @throws NullPointerException if `threadFactory` is `null`
     */
    /**
     * Creates a new timer with the default thread factory
     * ([Executors.defaultThreadFactory]), default tick duration, and
     * default number of ticks per wheel.
     */
    init {
        if (threadFactory == null) {
            throw NullPointerException("threadFactory")
        }
        if (unit == null) {
            throw NullPointerException("unit")
        }
        require(tickDuration > 0) { "tickDuration must be greater than 0: $tickDuration" }
        require(ticksPerWheel > 0) { "ticksPerWheel must be greater than 0: $ticksPerWheel" }

        // Normalize ticksPerWheel to power of two and initialize the wheel.
        wheel = createWheel(ticksPerWheel)
        mask = wheel.size - 1

        // Convert tickDuration to nanos.
        this.tickDuration = unit.toNanos(tickDuration)

        // Prevent overflow.
        require(this.tickDuration < Long.MAX_VALUE / wheel.size) {
            String.format(
                "tickDuration: %d (expected: 0 < tickDuration in nanos < %d",
                tickDuration,
                Long.MAX_VALUE / wheel.size
            )
        }
        workerThread = threadFactory.newThread(worker)
        this.maxPendingTimeouts = maxPendingTimeouts
        if (instanceCounter.incrementAndGet() > INSTANCE_COUNT_LIMIT
            && warnedTooManyInstances.compareAndSet(false, true)
        ) {
            reportTooManyInstances()
        }
    }

    @Suppress("removal")
    @Throws(Throwable::class)
    protected fun finalize() {
        try {
//            super.finalize()
        } finally {
            // This object is going to be GCed and it is assumed the ship has sailed to do a proper shutdown. If
            // we have not yet shutdown then we want to make sure we decrement the active instance count.
            if (workerStateUpdater.getAndSet(this, WORKER_STATE_SHUTDOWN) != WORKER_STATE_SHUTDOWN) {
                instanceCounter.decrementAndGet()
            }
        }
    }

    /**
     * Starts the background thread explicitly.  The background thread will
     * start automatically on demand even if you did not call this method.
     *
     * @throws IllegalStateException if this timer has been
     * [stopped][.stop] already
     */
    fun start() {
        when (workerStateUpdater[this]) {
            WORKER_STATE_INIT -> if (workerStateUpdater.compareAndSet(this, WORKER_STATE_INIT, WORKER_STATE_STARTED)) {
                workerThread.start()
            }

            WORKER_STATE_STARTED -> {}
            WORKER_STATE_SHUTDOWN -> throw IllegalStateException("cannot be started once stopped")
            else -> throw Error("Invalid WorkerState")
        }

        // Wait until the startTime is initialized by the worker.
        while (startTime == 0L) {
            try {
                startTimeInitialized.await()
            } catch (ignore: InterruptedException) {
                // Ignore - it will be ready very soon.
            }
        }
    }

    override fun stop(): Set<Timeout> {
        check(Thread.currentThread() !== workerThread) {
            (HashedWheelTimer::class.java.simpleName + ".stop() cannot be called from "
                    + TimerTask::class.java.simpleName)
        }
        if (!workerStateUpdater.compareAndSet(this, WORKER_STATE_STARTED, WORKER_STATE_SHUTDOWN)) {
            // workerState can be 0 or 2 at this moment - let it always be 2.
            if (workerStateUpdater.getAndSet(this, WORKER_STATE_SHUTDOWN) != WORKER_STATE_SHUTDOWN) {
                instanceCounter.decrementAndGet()
            }
            return emptySet()
        }
        try {
            var interrupted = false
            while (workerThread.isAlive) {
                workerThread.interrupt()
                try {
                    workerThread.join(100)
                } catch (ignored: InterruptedException) {
                    interrupted = true
                }
            }
            if (interrupted) {
                Thread.currentThread().interrupt()
            }
        } finally {
            instanceCounter.decrementAndGet()
        }
        return worker.unprocessedTimeouts()
    }

    override fun newTimeout(task: TimerTask, delay: Long, unit: TimeUnit): Timeout {
        if (task == null) {
            throw NullPointerException("task")
        }
        if (unit == null) {
            throw NullPointerException("unit")
        }
        val pendingTimeoutsCount = pendingTimeouts.incrementAndGet()
        if (maxPendingTimeouts > 0 && pendingTimeoutsCount > maxPendingTimeouts) {
            pendingTimeouts.decrementAndGet() // 这里的顺序调整一下不是更好 ...
            throw RejectedExecutionException(
                "Number of pending timeouts (" + pendingTimeoutsCount
                        + ") is greater than or equal to maximum allowed pending "
                        + "timeouts (" + maxPendingTimeouts + ")"
            )
        }
        start()

        // Add the timeout to the timeout queue which will be processed on the next tick.
        // During processing all the queued HashedWheelTimeouts will be added to the correct HashedWheelBucket.
        var deadline = System.nanoTime() + unit.toNanos(delay) - startTime

        // Guard against overflow.
        if (delay > 0 && deadline < 0) {
            deadline = Long.MAX_VALUE
        }
        val timeout = HashedWheelTimeout(this, task, deadline)
        timeouts.add(timeout)
        return timeout
    }

    /**
     * Returns the number of pending timeouts of this [Timer].
     */
    fun pendingTimeouts(): Long {
        return pendingTimeouts.get()
    }

    private inner class Worker : Runnable {
        private val unprocessedTimeouts: MutableSet<Timeout> = HashSet()
        private var tick: Long = 0
        override fun run() {
            // Initialize the startTime.
            startTime = System.nanoTime()
            if (startTime == 0L) {
                // We use 0 as an indicator for the uninitialized value here, so make sure it's not 0 when initialized.
                startTime = 1
            }

            // Notify the other threads waiting for the initialization at start().
            startTimeInitialized.countDown()
            do {
                val deadline = waitForNextTick()
                if (deadline > 0) {
                    val idx = (tick and mask.toLong()).toInt()
                    processCancelledTasks() // 1. 处理要取消的人呢五
                    val bucket = wheel[idx] // 2. 确定要搞的 bucket
                    transferTimeoutsToBuckets() // 3. 有一些任务 可以加入到对应的 bucket， 一次最多进去 10w 个、硬编码写死了
                    bucket!!.expireTimeouts(deadline) // 4. 这里应该会执行对应 bucket 的任务
                    tick++
                }
            } while (workerStateUpdater[this@HashedWheelTimer] == WORKER_STATE_STARTED)

            // Fill the unprocessedTimeouts so we can return them from stop() method.
            for (bucket in wheel) {
                bucket!!.clearTimeouts(unprocessedTimeouts)
            }
            while (true) {
                val timeout = timeouts.poll() ?: break
                if (!timeout.isCancelled) {
                    unprocessedTimeouts.add(timeout)
                }
            }
            processCancelledTasks()
        }

        private fun transferTimeoutsToBuckets() {
            // transfer only max. 100000 timeouts per tick to prevent a thread to stale the workerThread when it just
            // adds new timeouts in a loop.
            for (i in 0..99999) {
                val timeout = timeouts.poll()
                    ?: // all processed
                    break
                if (timeout.state() == HashedWheelTimeout.ST_CANCELLED) {
                    // Was cancelled in the meantime.
                    continue
                }
                val calculated = timeout.deadline / tickDuration
                timeout.remainingRounds = (calculated - tick) / wheel.size
                val ticks = Math.max(calculated, tick) // Ensure we don't schedule for past.
                val stopIndex = (ticks and mask.toLong()).toInt()
                val bucket = wheel[stopIndex]
                bucket!!.addTimeout(timeout)
            }
        }

        private fun processCancelledTasks() {
            while (true) {
                val timeout = cancelledTimeouts.poll()
                    ?: // all processed
                    break
                try {
                    timeout.remove()
                } catch (t: Throwable) {
                    if (LOG.isWarnEnabled) {
                        LOG.warn("An exception was thrown while process a cancellation task", t)
                    }
                }
            }
        }

        /**
         * Calculate goal nanoTime from startTime and current tick number,
         * then wait until that goal has been reached.
         *
         * @return Long.MIN_VALUE if received a shutdown request,
         * current time otherwise (with Long.MIN_VALUE changed by +1)
         */
        private fun waitForNextTick(): Long {
            val deadline = tickDuration * (tick + 1)
            while (true) {
                val currentTime = System.nanoTime() - startTime
                val sleepTimeMs = (deadline - currentTime + 999999) / 1000000
                if (sleepTimeMs <= 0) {
                    return if (currentTime == Long.MIN_VALUE) {
                        -Long.MAX_VALUE
                    } else {
                        currentTime
                    }
                }

                // We decide to remove the original approach (as below) which used in netty for
                // windows platform.
                // See https://github.com/netty/netty/issues/356
                //
                // if (Platform.isWindows()) {
                //     sleepTimeMs = sleepTimeMs / 10 * 10;
                // }
                //
                // The above approach that make sleepTimes to be a multiple of 10ms will
                // lead to severe spin in this loop for several milliseconds, which
                // causes the high CPU usage.
                // See https://github.com/sofastack/sofa-jraft/issues/311
                //
                // According to the regression testing on windows, we haven't reproduced the
                // Thread.sleep() bug referenced in https://www.javamex.com/tutorials/threads/sleep_issues.shtml
                // yet.
                //
                // The regression testing environment:
                // - SOFAJRaft version: 1.2.6
                // - JVM version (e.g. java -version): JDK 1.8.0_191
                // - OS version: Windows 7 ultimate 64 bit
                // - CPU: Intel(R) Core(TM) i7-2670QM CPU @ 2.20GHz (4 cores)
                try {
                    Thread.sleep(sleepTimeMs)
                } catch (ignored: InterruptedException) {
                    if (workerStateUpdater[this@HashedWheelTimer] == WORKER_STATE_SHUTDOWN) {
                        return Long.MIN_VALUE
                    }
                }
            }
        }

        fun unprocessedTimeouts(): Set<Timeout> {
            return Collections.unmodifiableSet(unprocessedTimeouts)
        }
    }

    private class HashedWheelTimeout internal constructor(
        val timer: HashedWheelTimer,
        private val task: TimerTask,
        val deadline: Long
    ) : Timeout {
        @Suppress("unused")
        @Volatile
        private var state = ST_INIT

        // remainingRounds will be calculated and set by Worker.transferTimeoutsToBuckets() before the
        // HashedWheelTimeout will be added to the correct HashedWheelBucket.
        var remainingRounds: Long = 0

        // This will be used to chain timeouts in HashedWheelTimerBucket via a double-linked-list.
        // As only the workerThread will act on it there is no need for synchronization / volatile.
        var next: HashedWheelTimeout? = null
        var prev: HashedWheelTimeout? = null

        // The bucket to which the timeout was added
        var bucket: HashedWheelBucket? = null
        override fun timer(): Timer {
            return timer
        }

        override fun task(): TimerTask {
            return task
        }

        override fun cancel(): Boolean {
            // only update the state it will be removed from HashedWheelBucket on next tick.
            if (!compareAndSetState(ST_INIT, ST_CANCELLED)) {
                return false
            }
            // If a task should be canceled we put this to another queue which will be processed on each tick.
            // So this means that we will have a GC latency of max. 1 tick duration which is good enough. This way
            // we can make again use of our MpscLinkedQueue and so minimize the locking / overhead as much as possible.
            timer.cancelledTimeouts.add(this)
            return true
        }

        fun remove() {
            val bucket = bucket
            if (bucket != null) {
                bucket.remove(this)
            } else {
                timer.pendingTimeouts.decrementAndGet()
            }
        }

        fun compareAndSetState(expected: Int, state: Int): Boolean {
            return STATE_UPDATER.compareAndSet(this, expected, state)
        }

        fun state(): Int {
            return state
        }

        override val isCancelled: Boolean
            get() = state() == ST_CANCELLED
        override val isExpired: Boolean
            get() = state() == ST_EXPIRED

        fun expire() {
            if (!compareAndSetState(ST_INIT, ST_EXPIRED)) {
                return
            }
            try {
                task.run(this)
            } catch (t: Throwable) {
                if (LOG.isWarnEnabled) {
                    LOG.warn("An exception was thrown by " + TimerTask::class.java.simpleName + '.', t)
                }
            }
        }

        override fun toString(): String {
            val currentTime = System.nanoTime()
            val remaining = deadline - currentTime + timer.startTime
            val buf = StringBuilder(192).append(javaClass.simpleName).append('(')
                .append("deadline: ")
            if (remaining > 0) {
                buf.append(remaining).append(" ns later")
            } else if (remaining < 0) {
                buf.append(-remaining).append(" ns ago")
            } else {
                buf.append("now")
            }
            if (isCancelled) {
                buf.append(", cancelled")
            }
            return buf.append(", task: ").append(task()).append(')').toString()
        }

        companion object {
            private const val ST_INIT = 0
            const val ST_CANCELLED = 1
            private const val ST_EXPIRED = 2
            private val STATE_UPDATER = AtomicIntegerFieldUpdater
                .newUpdater(
                    HashedWheelTimeout::class.java,
                    "state"
                )
        }
    }

    /**
     * Bucket that stores HashedWheelTimeouts. These are stored in a linked-list like datastructure to allow easy
     * removal of HashedWheelTimeouts in the middle. Also the HashedWheelTimeout act as nodes themself and so no
     * extra object creation is needed.
     */
    private class HashedWheelBucket {
        // Used for the linked-list datastructure
        private var head: HashedWheelTimeout? = null
        private var tail: HashedWheelTimeout? = null

        /**
         * Add [HashedWheelTimeout] to this bucket.
         */
        fun addTimeout(timeout: HashedWheelTimeout) {
            assert(timeout.bucket == null)
            timeout.bucket = this
            if (head == null) {
                tail = timeout
                head = tail
            } else {
                tail!!.next = timeout
                timeout.prev = tail
                tail = timeout
            }
        }

        /**
         * Expire all [HashedWheelTimeout]s for the given `deadline`.
         */
        fun expireTimeouts(deadline: Long) {
            var timeout = head

            // process all timeouts
            while (timeout != null) {
                var next = timeout.next
                if (timeout.remainingRounds <= 0) {
                    next = remove(timeout)
                    if (timeout.deadline <= deadline) {
                        timeout.expire()
                    } else {
                        // The timeout was placed into a wrong slot. This should never happen.
                        throw IllegalStateException(
                            String.format(
                                "timeout.deadline (%d) > deadline (%d)",
                                timeout.deadline, deadline
                            )
                        )
                    }
                } else if (timeout.isCancelled) {
                    next = remove(timeout)
                } else {
                    timeout.remainingRounds--
                }
                timeout = next
            }
        }

        fun remove(timeout: HashedWheelTimeout): HashedWheelTimeout? {
            val next = timeout.next
            // remove timeout that was either processed or cancelled by updating the linked-list
            if (timeout.prev != null) {
                timeout.prev!!.next = next
            }
            if (timeout.next != null) {
                timeout.next!!.prev = timeout.prev
            }
            if (timeout == head) {
                // if timeout is also the tail we need to adjust the entry too
                if (timeout == tail) {
                    tail = null
                    head = null
                } else {
                    head = next
                }
            } else if (timeout == tail) {
                // if the timeout is the tail modify the tail to be the prev node.
                tail = timeout.prev
            }
            // null out prev, next and bucket to allow for GC.
            timeout.prev = null
            timeout.next = null
            timeout.bucket = null
            timeout.timer.pendingTimeouts.decrementAndGet()
            return next
        }

        /**
         * Clear this bucket and return all not expired / cancelled [Timeout]s.
         */
        fun clearTimeouts(set: MutableSet<Timeout>) {
            while (true) {
                val timeout = pollTimeout() ?: return
                if (timeout.isExpired || timeout.isCancelled) {
                    continue
                }
                set.add(timeout)
            }
        }

        private fun pollTimeout(): HashedWheelTimeout? {
            val head = head ?: return null
            val next = head.next
            if (next == null) {
                this.head = null
                tail = this.head
            } else {
                this.head = next
                next.prev = null
            }

            // null out prev and next to allow for GC.
            head.next = null
            head.prev = null
            head.bucket = null
            return head
        }
    }

    companion object {
        private val LOG = LoggerFactory
            .getLogger(HashedWheelTimer::class.java)
        private const val INSTANCE_COUNT_LIMIT = 4
        private val instanceCounter = AtomicInteger()
        private val warnedTooManyInstances = AtomicBoolean()
        private val workerStateUpdater = AtomicIntegerFieldUpdater
            .newUpdater(
                HashedWheelTimer::class.java,
                "workerState"
            )
        const val WORKER_STATE_INIT = 0
        const val WORKER_STATE_STARTED = 1
        const val WORKER_STATE_SHUTDOWN = 2
        private fun createWheel(ticksPerWheel: Int): Array<HashedWheelBucket?> {
            var ticksPerWheel = ticksPerWheel
            require(ticksPerWheel > 0) { "ticksPerWheel must be greater than 0: $ticksPerWheel" }
            require(ticksPerWheel <= 1073741824) { "ticksPerWheel may not be greater than 2^30: $ticksPerWheel" }
            ticksPerWheel = normalizeTicksPerWheel(ticksPerWheel)
            val wheel = arrayOfNulls<HashedWheelBucket>(ticksPerWheel)
            for (i in wheel.indices) {
                wheel[i] = HashedWheelBucket()
            }
            return wheel
        }

        private fun normalizeTicksPerWheel(ticksPerWheel: Int): Int {
            var normalizedTicksPerWheel = 1
            while (normalizedTicksPerWheel < ticksPerWheel) {
                normalizedTicksPerWheel = normalizedTicksPerWheel shl 1
            }
            return normalizedTicksPerWheel
        }

        private fun reportTooManyInstances() {
            val resourceType = HashedWheelTimer::class.java.simpleName
            LOG.error(
                "You are creating too many {} instances.  {} is a shared resource that must be "
                        + "reused across the JVM, so that only a few instances are created.", resourceType,
                resourceType
            )
        }
    }
}