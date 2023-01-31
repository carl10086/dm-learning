package com.ysz.dm.ysz.base.mysql.config.dynamic

import org.slf4j.LoggerFactory

/**
 *<pre>
 * class desc here
 *</pre>
 *@author carl.yu
 *@since 2023/1/31
 **/
object DynamicDataSourceHolder {
    private val holder = ThreadLocal<Boolean>()

    fun forcePrimary(): Unit = holder.set(true)

    fun get() = if (this.holder.get() == true) DynamicDataSourceType.PRIMARY else DynamicDataSourceType.SECONDARY

    fun reset(): Unit = holder.remove()

}

enum class DynamicDataSourceType {
    PRIMARY,
    SECONDARY
}