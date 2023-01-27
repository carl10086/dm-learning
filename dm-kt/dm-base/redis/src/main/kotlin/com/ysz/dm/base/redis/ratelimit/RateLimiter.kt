package com.ysz.dm.base.redis.ratelimit

/**
 * @author carl
 * @create 2022-12-08 5:48 PM
 **/
interface RateLimiter<REQ> {

    /**
     * receive one req
     */
    fun recv(req: REQ): RateLimitResp
}