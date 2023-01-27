package com.ysz.dm.base.redis.ratelimit

import com.ysz.dm.base.redis.lua.LettuceEvalTools
import com.ysz.dm.base.redis.lua.RedisLuaScript
import io.lettuce.core.ScriptOutputType
import io.lettuce.core.api.sync.RedisCommands
import org.apache.commons.io.IOUtils
import java.nio.charset.StandardCharsets
import java.time.Duration
import kotlin.math.abs

/**
 * <pre>
 *     概述:
 *     1. 和 固定窗口算法类似
 *     2. 在当前窗口不超过 limit 的时候, 会按照比例计算前1个窗口的值, 和当前窗口的值 , 例如 0.7 * prevWindowNum + 0.3 * curWindowNum 做一个近似的计算, 用这个值来判断.
 *      2.1 这个 0.7 怎么算出来的, 是基于 当前的 时间在当前窗口过去的时间 百分比来算的, 比如说当前过去了 30% 的时间, 就会把之前的窗口当做 70% 的成分来计算
 * </pre>
 * @author carl
 * @create 2022-12-08 9:23 PM
 **/
class DoubleWindowRateLimiter(private val sync: RedisCommands<String, String>) : RateLimiter<DoubleWindowReq> {
    override fun recv(req: DoubleWindowReq): RateLimitResp {
        val occurAtMills = req.occurAt
        val currentWindow: Long = occurAtMills / req.timeWindow.toMillis()
        val fullKey = "${req.key}:${currentWindow}"


        /*4 * timeWindows ttl, prev window also need to keep for more time*/
        val ttl = req.timeWindow.toSeconds().toInt() * 4

        val count: Long = LettuceEvalTools.evalLuaScript(
            lua,
            sync,
            ScriptOutputType.INTEGER,
            arrayOf(),
            "${req.key}:",
            currentWindow.toString(),
            ttl.toString(),
            req.maxReq.toString(),
//            req.percentageOfPrevWindow.toString()
            calPercentage(req.occurAt, req.timeWindow).toString()
        )

        return RateLimitResp(count >= 0, abs(count), fullKey)
    }


    private fun calPercentage(occurAt: Long, timeWindow: Duration): Float {
        return 1.0f - (occurAt % timeWindow.toMillis()).toFloat() / (timeWindow.toMillis().toFloat())
    }

    companion object {
        private val lua: RedisLuaScript = RedisLuaScript(
            IOUtils.toString(
                FixedWindowRateLimiter::class.java.getResource("/dm/lua/ratelimit/double_window.lua"),
                StandardCharsets.UTF_8
            )
        )
    }
}