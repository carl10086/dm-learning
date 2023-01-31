package com.ysz.dm.ysz.base.mysql.config.dynamic

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

/**
 * <pre>
 * class desc here
</pre> *
 * @author carl.yu
 * @since 2023/1/31
 */
internal class DynamicDataSourceHolderTest {
    @Test
    fun `test get`() {
        assertEquals(DynamicDataSourceHolder.get(), DynamicDataSourceType.SECONDARY)

        DynamicDataSourceHolder.forcePrimary()
        assertEquals(DynamicDataSourceHolder.get(), DynamicDataSourceType.PRIMARY)

        DynamicDataSourceHolder.reset()
        assertEquals(DynamicDataSourceHolder.get(), DynamicDataSourceType.SECONDARY)
    }
}
