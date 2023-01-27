package com.ysz.dm.lib.lang.juc

import com.ysz.dm.lib.lang.juc.custom.CustomScheduledThreadPoolExecutor
import org.junit.jupiter.api.Test
import org.slf4j.LoggerFactory

/**
 * @author carl
 * @create 2022-11-28 4:57 PM
 **/
class ScheduledThreadPoolExecutorTest {

    @Test
    internal fun `test_hello`() {

        val pool = CustomScheduledThreadPoolExecutor(3)
        pool.submit {
            logger.info("runnable")
        }

        pool.shutdown()

    }

    companion object {
        private val logger = LoggerFactory.getLogger(ScheduledThreadPoolExecutorTest::class.java)
    }
}