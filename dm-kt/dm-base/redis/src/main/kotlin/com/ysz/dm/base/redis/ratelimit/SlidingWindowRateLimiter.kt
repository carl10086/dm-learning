package com.ysz.dm.base.redis.ratelimit

import com.ysz.dm.base.redis.lua.LettuceEvalTools
import com.ysz.dm.base.redis.lua.RedisLuaScript
import io.lettuce.core.ScriptOutputType
import io.lettuce.core.api.sync.RedisCommands
import org.apache.commons.io.IOUtils
import org.slf4j.LoggerFactory
import java.nio.charset.StandardCharsets
import kotlin.math.abs

/**
 * <pre>
 *     一个优化过的滑动窗口限流 .
 *
 *     思路如下:
 *     1. 举个例子: 假设是 10min 的窗口,分为10个 units, 从 第1min 开始.
 *
 *     2. 在 第 5min 的时候:
 *     - 5min 这个窗口第1次收到请求的时候, 会计算前面9个窗口的总和  记录为 base,注意,这次计算会比较伤性能 . 因为要算之前的所有窗口
 *     - 后面 每个请求, 基于第一次计算的 base + 当前窗口对应的 window 的总和 来进行判断
 *
 *     lua 脚本入参:
 *     1. maxReq : 限流的最大值
 *     2. key 前缀: 会加上 + ":"
 *     3. 当前窗口: currentWindow
 *     4. 要被移除的最开始的窗口, 还是上面的例子, 如果是 当前11min , 就要移除 1min窗口.
 *     5. ttl for key
 *
 *
 *     返回值:
 *     1. 一个数字 大于0 成功 小于0 失败,  都代表了一个有意义的计数
 *
 *
 *
 * </pre>
 * @author carl
 * @create 2022-12-08 5:43 PM
 **/
class SlidingWindowRateLimiter(private val sync: RedisCommands<String, String>) : RateLimiter<SlidingWindowReq> {
    override fun recv(req: SlidingWindowReq): RateLimitResp {


        val occurAtMills = req.occurAt
        val currentWindow: Long = occurAtMills / req.timeWindowUnit.toMillis()

        /* + 2 */
        val ttl = req.timeWindowUnit.toSeconds().toInt() * (req.unitNum + 2)

//        val toBeRmTimeWindow = "${req.key}:${currentWindow - req.unitNum}"

        try {
            val count = LettuceEvalTools.evalLuaScript<String, String, Long>(
                lua,
                sync,
                ScriptOutputType.INTEGER,
                arrayOf(),
                req.maxReq.toString(),
                req.key + ":",
                currentWindow.toString(),
                req.unitNum.toString(),
                ttl.toString()
            )
            return RateLimitResp(count >= 0, abs(count), "${req.key}:${currentWindow}")
        } catch (e: RuntimeException) {
            log.error("script error, curr:${currentWindow}")
            throw e
        }

    }


    companion object {

        private val log = LoggerFactory.getLogger(SlidingWindowRateLimiter::class.java)
        private val lua: RedisLuaScript = RedisLuaScript(
            IOUtils.toString(
                FixedWindowRateLimiter::class.java.getResource("/dm/lua/ratelimit/sliding_window.lua"),
                StandardCharsets.UTF_8
            )
        )
    }
}