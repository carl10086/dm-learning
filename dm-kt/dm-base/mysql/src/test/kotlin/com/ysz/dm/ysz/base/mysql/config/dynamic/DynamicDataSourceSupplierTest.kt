package com.ysz.dm.ysz.base.mysql.config.dynamic

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import java.util.concurrent.CompletableFuture

/**
 * <pre>
 * class desc here
</pre> *
 * @author carl.yu
 * @since 2023/1/31
 */
internal class DynamicDataSourceSupplierTest {
    @Test
    fun `test get`() {
        /*force in the primary*/
        DynamicDataSourceHolder.forcePrimary()

        assertEquals(CompletableFuture.supplyAsync(
            DynamicDataSourceSupplier gs{
                /*return the child thread*/
                DynamicDataSourceHolder.get()
            }
        ).apply { this.join() }.get(), DynamicDataSourceType.PRIMARY
        )

        /*force in the primary*/
        DynamicDataSourceHolder.reset()

        assertEquals(CompletableFuture.supplyAsync(
            DynamicDataSourceSupplier {
                /*return the child thread*/
                DynamicDataSourceHolder.get()
            }
        ).apply { this.join() }.get(), DynamicDataSourceType.SECONDARY
        )


    }
}