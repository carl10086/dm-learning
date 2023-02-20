package com.ysz.dm.base.repo.impl.jdbctpl

import com.ysz.dm.base.repo.repository.Repository
import com.ysz.dm.base.repo.repository.RepositoryMetaV1
import com.ysz.dm.base.repo.support.mapping.PropertyColNameConverter
import org.springframework.jdbc.core.JdbcTemplate

/**
 * @author carl
 * @since 2023-02-17 3:39 AM
 **/
class JdbcTemplateFactory(
    private val jdbcTemplate: JdbcTemplate
) {
    fun <T : Any, ID, I : Repository<T, ID>> makeForTable(
        table: String,
        infJavaClz: Class<I>,
        converter: PropertyColNameConverter = PropertyColNameConverter.CAMEL_TO_UNDERSCORE
    ): I {
        return JdbcTemplateProxy(
            infJavaClz,
            SimpleJdbcRepository(
                RepositoryMetaV1(infJavaClz),
                table,
                converter,
                this.jdbcTemplate
            )
        ).getProxy()
    }
}