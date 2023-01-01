package com.ysz.dm.base.redis.ratelimit

import java.time.Duration

/**
 * @author carl
 * @create 2022-12-08 9:23 PM
 **/
data class DoubleWindowReq(
    /*时间窗口大小*/
    val timeWindow: Duration,
    /*单位时间窗口中最大的允许的量*/
    val maxReq: Int,
    /*毫秒时间戳*/
    val occurAt: Long,
    val key: String,
    /**
     * 上一个窗口的百分比
     */
    val percentageOfPrevWindow: Float = 0.7f
) {
}