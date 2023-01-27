package com.ysz.dm.base.core.timer

interface TimerTask {
    /**
     * Executed after the delay specified with
     * Timer#newTimeout(TimerTask, long, TimeUnit).
     *
     * @param timeout a handle which is associated with this task
     */
    @Throws(Exception::class)
    fun run(timeout: Timeout)
}