package com.ysz.dm.base.hbase

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.hbase.HBaseConfiguration
import org.apache.hadoop.hbase.client.Connection
import org.apache.hadoop.hbase.client.ConnectionFactory
import org.slf4j.LoggerFactory
import org.springframework.beans.factory.DisposableBean
import org.springframework.beans.factory.FactoryBean
import org.springframework.beans.factory.InitializingBean

/**
 * @author carl
 * @since 2022-12-29 9:49 PM
 **/
open class HBaseConnectionBean(
    val classPathResource: String = "conf/hbase-site.xml"
) : FactoryBean<Connection>, InitializingBean, DisposableBean {

    private lateinit var conn: Connection

    override fun getObject(): Connection {
        return this.conn
    }

    override fun getObjectType(): Class<*> {
        return Connection::class.java
    }

    override fun afterPropertiesSet() {
        val conf = HBaseConfiguration.create(Configuration().apply {
            addResource(
                Thread.currentThread().contextClassLoader.getResource(
                    "conf/hbase-site.xml"
                )
            )
        }
        )

        this.conn = ConnectionFactory.createConnection(conf)

        log.info("HBaseConnectionBean init")
    }

    override fun destroy() {
        conn.close()
        log.info("HBaseConnectionBean closed")
    }

    companion object {
        private val log = LoggerFactory.getLogger(HBaseConnectionBean::class.java)
    }

}