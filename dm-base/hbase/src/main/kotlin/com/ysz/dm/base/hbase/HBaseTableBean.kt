package com.ysz.dm.base.hbase

import org.apache.hadoop.hbase.TableName
import org.apache.hadoop.hbase.client.Connection
import org.apache.hadoop.hbase.client.HTable
import org.apache.hadoop.hbase.client.Table
import org.slf4j.LoggerFactory
import org.springframework.beans.factory.DisposableBean
import org.springframework.beans.factory.FactoryBean
import org.springframework.beans.factory.InitializingBean


/**
 * @author carl
 * @since 2022-12-29 9:54 PM
 **/
open class HBaseTableBean(
    private val tableName: TableName,
    private val connection: Connection
) : FactoryBean<Table>, InitializingBean, DisposableBean {

    private lateinit var hTable: Table

    override fun getObject(): Table {
        return this.hTable
    }

    override fun getObjectType(): Class<*> {
        return HTable::class.java
    }

    override fun afterPropertiesSet() {
        this.hTable = connection.getTable(tableName)
        log.info("hbase table init success")
    }

    override fun destroy() {
        log.info("hbase table close success")
        this.hTable.close()
    }

    companion object {
        private val log = LoggerFactory.getLogger(HBaseConnectionBean::class.java)
    }
}