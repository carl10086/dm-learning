package com.ysz.dm.rb.base.core.tools.json

import com.fasterxml.jackson.module.kotlin.*
import org.junit.jupiter.api.Test
import org.slf4j.LoggerFactory

/**
 *<pre>
 * class desc here
 *</pre>
 *@author carl.yu
 *@createAt 2022/11/13
 **/
internal class JsonToolsTest {
    private val mapper = JsonTools.mapper


    @Test
    internal fun `test_json_user`() {
        var json = mapper.writeValueAsString(User())
        log.info("user to json string:{}", json)
        log.info("json string to user:{}", mapper.readValue<User>(json))


        json = mapper.writeValueAsString(listOf(User(), User("bbb")))
        log.info("user to json string:{}", json)
        log.info("json string to user:{}", mapper.readValue<List<User>>(json))

    }

    data class User(val username: String = "aaa", val age: Int = 10)

    companion object {
        val log = LoggerFactory.getLogger(JsonToolsTest::class.java)
    }
}