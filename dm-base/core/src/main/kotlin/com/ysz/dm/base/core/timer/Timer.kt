package com.ysz.dm.base.core.timer

import java.util.concurrent.TimeUnit


interface Timer {
    /**
     * Schedules the specified [TimerTask] for one-time execution after
     * the specified delay.
     *
     * @return a handle which is associated with the specified task
     *
     * @throws IllegalStateException       if this timer has been [stopped][.stop] already
     * @throws java.util.concurrent.RejectedExecutionException if the pending timeouts are too many and creating new timeout
     * can cause instability in the system.
     */
    fun newTimeout(task: TimerTask, delay: Long, unit: TimeUnit): Timeout

    /**
     * Releases all resources acquired by this [Timer] and cancels all
     * tasks which were scheduled but not executed yet.
     *
     * @return the handles associated with the tasks which were canceled by
     * this method
     */
    fun stop(): Set<Timeout>
}