package com.ysz.dm.rb.it.ratelimit

import org.junit.jupiter.api.Test
import org.slf4j.LoggerFactory
import java.time.Duration
import java.util.*

/**
 * @author carl
 * @create 2022-11-21 12:26 PM
 **/
internal class RateLimitTest {

    fun fixWindow(
        key: String, interval: Duration, maxReqs: Int
    ): Boolean {
        return true
    }

    @Test
    internal fun `test_fixWindow`() {
        val interval = Duration.ofMinutes(10L)
        val fixWindowKey = fixWindowKey(interval)
        if (log.isDebugEnabled) {
            log.debug("window:{}, As time:{}", fixWindowKey, Date(interval.toMillis() * fixWindowKey))
        }
    }

    private fun fixWindowKey(interval: Duration) = System.currentTimeMillis() / 1000L / interval.toSeconds()

    companion object {
        private val log = LoggerFactory.getLogger(RateLimitTest::class.java)
    }
}