package com.ysz.dm.base.redis.ratelimit

import com.ysz.dm.base.redis.lua.LettuceEvalTools
import com.ysz.dm.base.redis.lua.RedisLuaScript
import io.lettuce.core.ScriptOutputType
import io.lettuce.core.api.sync.RedisCommands
import org.apache.commons.io.IOUtils
import java.nio.charset.StandardCharsets

/**
 * <pre>
 *     介绍:
 *     1. 类似滑动窗口的计数器, 不考虑 maxReq 这个参数,性能好.
 *     2. 主要为了兼容以前的业务 .
 *
 * </pre>
 * @author carl
 * @create 2022-12-08 5:31 PM
 **/
class FixedWindowCounter(private val sync: RedisCommands<String, String>) {

    fun recv(req: FixedWindowReq): RateLimitResp {
        val occurAtMills = req.occurAt
        val currentWindow: Long = occurAtMills / req.timeWindow.toMillis()
        val fullKey = "${req.key}:${currentWindow}"

        /*这里大方一点了, 给了2个时间窗口的 ttl*/
        val ttl = req.timeWindow.toSeconds().toInt() * 2



        return RateLimitResp(
            true,
            LettuceEvalTools.evalLuaScript(
                lua,
                sync,
                ScriptOutputType.INTEGER,
                arrayOf(fullKey),
                ttl.toString()
            ),
            fullKey
        )
    }


    companion object {
        private val lua: RedisLuaScript = RedisLuaScript(
            IOUtils.toString(
                FixedWindowRateLimiter::class.java.getResource("/dm/lua/ratelimit/fixed_window_counter.lua"),
                StandardCharsets.UTF_8
            )
        )
    }
}