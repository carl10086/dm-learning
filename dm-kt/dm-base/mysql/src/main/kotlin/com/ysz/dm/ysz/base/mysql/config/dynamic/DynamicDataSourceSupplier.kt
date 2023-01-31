package com.ysz.dm.ysz.base.mysql.config.dynamic

import java.util.function.Supplier

/**
 *<pre>
 * class desc here
 *</pre>
 *@author carl.yu
 *@since 2023/1/31
 **/
class DynamicDataSourceSupplier<V>(
    private val flag: DynamicDataSourceType = DynamicDataSourceHolder.get(),
    private val task: Supplier<V>,
) : Supplier<V> {
    override fun get(): V {
        return when (flag) {
            DynamicDataSourceType.PRIMARY -> {
                DynamicDataSourceHolder.forcePrimary()
                try {
                    return task.get()
                } finally {
                    DynamicDataSourceHolder.reset()
                }
            }

            else -> task.get()
        }
    }
}