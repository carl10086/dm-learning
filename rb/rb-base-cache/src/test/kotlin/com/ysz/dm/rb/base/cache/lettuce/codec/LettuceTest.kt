package com.ysz.dm.rb.base.cache.lettuce.codec

import io.lettuce.core.RedisClient
import io.lettuce.core.api.StatefulRedisConnection
import io.lettuce.core.api.sync.RedisCommands
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.slf4j.LoggerFactory
import redis.embedded.RedisServer

/**
 *<pre>
 * class desc here
 *</pre>
 *@author carl.yu
 *@createAt 2022/11/15
 **/
internal class LettuceTest {

    var server: RedisServer? = null

    @BeforeEach
    internal fun setUp() {
        server = RedisServer.builder().port(6379).build()
        server?.start()
    }

    @Test
    internal fun `test_get_set`() {
        val redisClient: RedisClient = RedisClient.create("redis://localhost:6379/0")
        val connection: StatefulRedisConnection<String, String> = redisClient.connect()
        val syncCommands: RedisCommands<String, String> = connection.sync()

        val value = "Hello, Redis!"
        syncCommands.set("key", value)
        Assertions.assertEquals(value, syncCommands.get("key"))

        connection.close()
        redisClient.shutdown()
    }

    @AfterEach
    internal fun tearDown() {
        server?.stop()
    }

    companion object {
        private val log = LoggerFactory.getLogger(LettuceTest::class.java)
    }
}