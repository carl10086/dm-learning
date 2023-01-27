package com.ysz.dm.base.core.timer

import org.slf4j.LoggerFactory
import java.util.concurrent.TimeUnit
import java.util.concurrent.locks.Lock
import java.util.concurrent.locks.ReentrantLock
import kotlin.jvm.internal.Intrinsics

abstract class RepeatedTimer(name: String, timeoutMs: Int, timer: Timer) {
    private val lock: Lock = ReentrantLock()
    private val timer: Timer
    private var timeout: Timeout? = null
    private var stopped: Boolean

    @Volatile
    private var running = false

    @Volatile
    private var destroyed = false

    @Volatile
    private var invoking = false

    @Volatile
    var timeoutMs: Int
        private set
    private val name: String

    init {
        Intrinsics.checkNotNull(timer)
        this.name = name
        this.timeoutMs = timeoutMs
        stopped = true
        this.timer = timer
    }

    /**
     * Subclasses should implement this method for timer trigger.
     */
    protected abstract fun onTrigger()

    /**
     * Adjust timeoutMs before every scheduling.
     *
     * @param timeoutMs timeout millis
     * @return timeout millis
     */
    protected fun adjustTimeout(timeoutMs: Int): Int {
        return timeoutMs
    }

    fun run() {
        invoking = true
        try {
            onTrigger()
        } catch (t: Throwable) {
            LOG.error("Run timer failed.", t)
        }
        var invokeDestroyed = false
        lock.lock()
        try {
            invoking = false
            if (stopped) {
                running = false
                invokeDestroyed = destroyed
            } else {
                timeout = null
                schedule()
            }
        } finally {
            lock.unlock()
        }
        if (invokeDestroyed) {
            onDestroy()
        }
    }

    /**
     * Run the timer at once, it will cancel the timer and re-schedule it.
     */
    fun runOnceNow() {
        lock.lock()
        try {
            if (timeout != null && timeout!!.cancel()) {
                timeout = null
                run()
            }
        } finally {
            lock.unlock()
        }
    }

    /**
     * Called after destroy timer.
     */
    protected fun onDestroy() {
        LOG.info("Destroy timer: {}.", this)
    }

    /**
     * Start the timer.
     */
    fun start() {
        lock.lock()
        try {
            if (destroyed) {
                return
            }
            if (!stopped) {
                return
            }
            stopped = false
            if (running) {
                return
            }
            running = true
            schedule()
        } finally {
            lock.unlock()
        }
    }

    /**
     * Restart the timer.
     * It will be started if it's stopped, and it will be restarted if it's running.
     *
     * @author Qing Wang (kingchin1218@gmail.com)
     *
     * 2020-Mar-26 20:38:37 PM
     */
    fun restart() {
        lock.lock()
        try {
            if (destroyed) {
                return
            }
            stopped = false
            running = true
            schedule()
        } finally {
            lock.unlock()
        }
    }

    private fun schedule() {
        if (timeout != null) {
            timeout!!.cancel()
        }
        val timerTask: TimerTask = object : TimerTask {
            override fun run(timeout: Timeout) {
                try {
                    this@RepeatedTimer.run()
                } catch (t: Throwable) {
                    LOG.error("Run timer task failed, taskName={}.", name, t)
                }
            }
        }
        timeout = timer
            .newTimeout(timerTask, adjustTimeout(timeoutMs).toLong(), TimeUnit.MILLISECONDS)
    }

    /**
     * Reset timer with new timeoutMs.
     *
     * @param timeoutMs timeout millis
     */
    fun reset(timeoutMs: Int) {
        lock.lock()
        this.timeoutMs = timeoutMs
        try {
            if (stopped) {
                return
            }
            if (running) {
                schedule()
            }
        } finally {
            lock.unlock()
        }
    }

    /**
     * Reset timer with current timeoutMs
     */
    fun reset() {
        lock.lock()
        try {
            reset(timeoutMs)
        } finally {
            lock.unlock()
        }
    }

    /**
     * Destroy timer
     */
    fun destroy() {
        var invokeDestroyed = false
        lock.lock()
        try {
            if (destroyed) {
                return
            }
            destroyed = true
            if (!running) {
                invokeDestroyed = true
            }
            if (stopped) {
                return
            }
            stopped = true
            if (timeout != null) {
                if (timeout!!.cancel()) {
                    invokeDestroyed = true
                    running = false
                }
                timeout = null
            }
        } finally {
            lock.unlock()
            timer.stop()
            if (invokeDestroyed) {
                onDestroy()
            }
        }
    }

    /**
     * Stop timer
     */
    fun stop() {
        lock.lock()
        try {
            if (stopped) {
                return
            }
            stopped = true
            if (timeout != null) {
                timeout!!.cancel()
                running = false
                timeout = null
            }
        } finally {
            lock.unlock()
        }
    }

    override fun toString(): String {
        return ("RepeatedTimer{" + "timeout=" + timeout + ", stopped=" + stopped + ", running="
                + running
                + ", destroyed=" + destroyed + ", invoking=" + invoking + ", timeoutMs="
                + timeoutMs
                + ", name='" + name + '\'' + '}')
    }

    companion object {
        val LOG = LoggerFactory.getLogger(RepeatedTimer::class.java)
    }
}