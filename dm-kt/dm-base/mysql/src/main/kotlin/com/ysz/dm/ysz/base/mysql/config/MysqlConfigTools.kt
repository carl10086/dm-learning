package com.ysz.dm.ysz.base.mysql.config

import com.zaxxer.hikari.HikariDataSource

/**
 *<pre>
 * class desc here
 *</pre>
 *@author carl.yu
 *@since 2023/1/31
 **/
object MysqlConfigTools {

    fun make(cfg: MysqlConfig): HikariDataSource {
        val ds = HikariDataSource()
        ds.driverClassName = cfg.driver
        ds.isAutoCommit = cfg.autoCommit
        ds.maximumPoolSize = cfg.max
        ds.minimumIdle = cfg.minIdle
        ds.isReadOnly = cfg.readOnly
        ds.jdbcUrl = cfg.jdbcUrl
        ds.username = cfg.username
        ds.password = cfg.password


        ds.addDataSourceProperty("cachePrepStmts", "true")
        ds.addDataSourceProperty("prepStmtCacheSize", "250")
        ds.addDataSourceProperty("prepStmtCacheSqlLimit", "2048")
        ds.addDataSourceProperty("useServerPrepStmts", "true")
        ds.addDataSourceProperty("useLocalSessionState", "true")
        ds.addDataSourceProperty("rewriteBatchedStatements", "true")
        ds.addDataSourceProperty("cacheResultSetMetadata", "true")
        ds.addDataSourceProperty("cacheServerConfiguration", "true")
        ds.addDataSourceProperty("elideSetAutoCommits", "true")
        ds.addDataSourceProperty("maintainTimeStats", "false")

        return ds
    }
}