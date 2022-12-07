package com.ysz.dm.rb.it.ratelimit

import io.lettuce.core.RedisClient
import io.lettuce.core.ScriptOutputType
import io.lettuce.core.api.StatefulRedisConnection
import io.lettuce.core.api.sync.RedisCommands

/**
 * @author carl
 * @create 2022-12-07 6:09 PM
 **/
internal class RedisLuaTst

fun main() {
    val redisClient: RedisClient = RedisClient.create("redis://localhost:6379/0")
    val connection: StatefulRedisConnection<String, String> = redisClient.connect()
    val syncCommands: RedisCommands<String, String> = connection.sync()

    val script = RedisLuaTst::class.java.getResource("/lua/test.lua").readText()
    println(script)
//    val scriptSha = syncCommands.scriptLoad(script)
//    println("$scriptSha")

    val result = syncCommands.evalsha<Long>(
        "9fff436e0a958dd7e0e02cc49f984aa6bb32dab4", ScriptOutputType.INTEGER, arrayOf("name"), "60", "5"
    )
    println(result)


}