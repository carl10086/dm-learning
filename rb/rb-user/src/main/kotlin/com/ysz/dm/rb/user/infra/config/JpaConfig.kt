package com.ysz.dm.rb.user.infra.config

import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import org.springframework.dao.annotation.PersistenceExceptionTranslationPostProcessor
import org.springframework.data.jpa.repository.config.EnableJpaRepositories
import org.springframework.jdbc.datasource.DriverManagerDataSource
import org.springframework.orm.jpa.JpaTransactionManager
import org.springframework.orm.jpa.LocalContainerEntityManagerFactoryBean
import org.springframework.orm.jpa.vendor.HibernateJpaVendorAdapter
import org.springframework.transaction.PlatformTransactionManager
import org.springframework.transaction.support.TransactionTemplate
import java.util.*
import javax.sql.DataSource

/**
 *<pre>
 * class desc here
 *</pre>
 *@author carl.yu
 *@createAt 2022/11/15
 **/
@Configuration
@EnableJpaRepositories(basePackages = ["com.ysz.dm.rb.user.infra.persist.dao"])
open class JpaConfig {

    private fun additionalProperties(): Properties {
        val hibernateProperties = Properties()
        hibernateProperties.setProperty("hibernate.hbm2ddl.auto", "create")
        hibernateProperties.setProperty("hibernate.dialect", "org.hibernate.dialect.H2Dialect")
        hibernateProperties.setProperty(
            "hibernate.physical_naming_strategy",
            "org.hibernate.boot.model.naming.CamelCaseToUnderscoresNamingStrategy"
        )
        return hibernateProperties
    }

    @Bean
    open fun dataSource(): DataSource? {
        val dataSource = DriverManagerDataSource()
        dataSource.setDriverClassName("org.h2.Driver")
        dataSource.url = "jdbc:h2:mem:myDb;DB_CLOSE_DELAY=-1"
        //    dataSource.setUsername(env.getProperty("jdbc.user"));
//    dataSource.setPassword(env.getProperty("jdbc.pass"));
        return dataSource
    }

    @Bean
    open fun entityManagerFactory(): LocalContainerEntityManagerFactoryBean? {
        val em = LocalContainerEntityManagerFactoryBean()
        em.dataSource = dataSource()
        em.setPackagesToScan("com.ysz.dm.rb.user.infra.persist.dataobject")
        val vendorAdapter = HibernateJpaVendorAdapter()
        em.jpaVendorAdapter = vendorAdapter
        em.setJpaProperties(additionalProperties())
        return em
    }

    @Bean
    open fun transactionManager(): PlatformTransactionManager? {
        val transactionManager = JpaTransactionManager()
        transactionManager.entityManagerFactory = entityManagerFactory()!!.getObject()
        return transactionManager
    }

    @Bean
    open fun transactionTemplate(): TransactionTemplate? {
        return TransactionTemplate(transactionManager())
    }

    @Bean
    open fun exceptionTranslation(): PersistenceExceptionTranslationPostProcessor? {
        return PersistenceExceptionTranslationPostProcessor()
    }


}

