package com.ysz.dm.base.redis.ratelimit

/**
 * @author carl
 * @create 2022-12-08 4:22 PM
 **/
data class RateLimitResp(
    /*true 代表成功*/
    val success: Boolean,
    /*当前窗口目前的计数, 方便调试*/
    val reqCountInCurrentWindow: Long,
    /*方便调试, 返回 redis key*/
    var key: String?
) {
}