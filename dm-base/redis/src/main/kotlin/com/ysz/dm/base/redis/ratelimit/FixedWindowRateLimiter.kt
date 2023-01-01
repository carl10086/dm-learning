package com.ysz.dm.base.redis.ratelimit

import com.ysz.dm.base.redis.lua.LettuceEvalTools
import com.ysz.dm.base.redis.lua.RedisLuaScript
import io.lettuce.core.ScriptOutputType
import io.lettuce.core.api.sync.RedisCommands
import org.apache.commons.io.IOUtils
import java.nio.charset.StandardCharsets
import kotlin.math.abs

/**
 * <pre>
 *    最简单的固定窗口 限流器 .
 *
 *    1.优点: 实现简单 ,正因为实现简单, 性能也基本 ok
 *
 *    2. 缺点: 无法应付 burst 流量. 这点不如其他的算法
 *      2.1 假设这样一种场景: 假设 1min 允许 100 个请求. 在 当前窗口的 55s-60s 刚好 100 个请求(之前没有) , 到了下一个窗口刚好 0s-5s 100个请求. 也 ok, 没有突破
 *          下场就是 10s 内qps = 200
 *
 *
 *     -  lua 脚本入参 ARGV 说明: 除了唯一的 KEYS[1] . 第1个入参是告诉 redis 在不存在的时候,设置 ttl, 第2个参数是限流的最大值, 超过则直接返回
 *     -  lua 脚本返回 : >=0 表示成功, 代表是当前窗口的计数, <0 表示失败 , 取正代表当前窗口的计数
 *
 *
 * </pre>
 * @author carl
 * @create 2022-12-08 3:52 PM
 **/
class FixedWindowRateLimiter(private val sync: RedisCommands<String, String>) : RateLimiter<FixedWindowReq> {

    override fun recv(req: FixedWindowReq): RateLimitResp {

        val occurAtMills = req.occurAt
        val currentWindow: Long = occurAtMills / req.timeWindow.toMillis()
        val fullKey = "${req.key}:${currentWindow}"

        /*这里大方一点了, 给了2个时间窗口的 ttl*/
        val ttl = req.timeWindow.toSeconds().toInt() * 2

        val count: Long = LettuceEvalTools.evalLuaScript(
            lua,
            sync,
            ScriptOutputType.INTEGER,
            arrayOf(fullKey),
            ttl.toString(),
            req.maxReq.toString()
        )


        return RateLimitResp(count >= 0, abs(count), fullKey)

    }


    companion object {
        private val lua: RedisLuaScript = RedisLuaScript(
            IOUtils.toString(
                FixedWindowRateLimiter::class.java.getResource("/dm/lua/ratelimit/fixed_window.lua"),
                StandardCharsets.UTF_8
            )
        )
    }

}