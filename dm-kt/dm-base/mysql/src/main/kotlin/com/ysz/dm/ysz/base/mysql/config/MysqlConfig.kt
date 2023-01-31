package com.ysz.dm.ysz.base.mysql.config

/**
 *<pre>
 * class desc here
 *</pre>
 *@author carl.yu
 *@since 2023/1/31
 **/
data class MysqlConfig(
    val jdbcUrl: String,
    val username: String,
    val password: String,
    val minIdle: Int = 16,
    val max: Int = 16,
    val maxWait: Int = 5000,
    val autoCommit: Boolean = true,
    val readOnly: Boolean = true,
    val driver: String = "com.mysql.cj.jdbc.Driver",
)