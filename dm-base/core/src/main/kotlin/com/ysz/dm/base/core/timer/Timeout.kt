package com.ysz.dm.base.core.timer

interface Timeout {
    /**
     * Returns the [Timer] that created this handle.
     */
    fun timer(): Timer

    /**
     * Returns the [TimerTask] which is associated with this handle.
     */
    fun task(): TimerTask

    /**
     * Returns `true` if and only if the [TimerTask] associated
     * with this handle has been expired.
     */
    val isExpired: Boolean

    /**
     * Returns `true` if and only if the [TimerTask] associated
     * with this handle has been cancelled.
     */
    val isCancelled: Boolean

    /**
     * Attempts to cancel the [TimerTask] associated with this handle.
     * If the task has been executed or cancelled already, it will return with
     * no side effect.
     *
     * @return True if the cancellation completed successfully, otherwise false
     */
    fun cancel(): Boolean
}