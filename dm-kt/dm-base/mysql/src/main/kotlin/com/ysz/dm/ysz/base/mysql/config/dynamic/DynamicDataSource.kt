package com.ysz.dm.ysz.base.mysql.config.dynamic

import org.springframework.jdbc.datasource.lookup.AbstractRoutingDataSource

/**
 *<pre>
 * class desc here
 *</pre>
 *@author carl.yu
 *@since 2023/1/31
 **/
open class DynamicDataSource : AbstractRoutingDataSource() {
    override fun determineCurrentLookupKey(): Any = DynamicDataSourceHolder.get()
}