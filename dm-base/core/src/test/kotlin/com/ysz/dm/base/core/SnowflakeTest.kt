package com.ysz.dm.base.core

import com.ysz.dm.base.core.tools.id.Snowflake
import org.junit.jupiter.api.Assertions

/**
 * <pre>
 * class desc here
</pre> *
 * @author carl.yu
 * @since 2022/12/24
 */
class SnowflakeTest {

    val snowflake = Snowflake()

    @org.junit.jupiter.api.Test
    fun nextId() {
        Assertions.assertTrue(snowflake.nextId() > 0L)

        // for 1_0000 times not the sample .
    }
}