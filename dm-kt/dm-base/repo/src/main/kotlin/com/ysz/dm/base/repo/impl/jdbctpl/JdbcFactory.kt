package com.ysz.dm.base.repo.impl.jdbctpl

import com.ysz.dm.base.repo.factory.RepositoryFactory
import com.ysz.dm.base.repo.repository.Repository
import org.springframework.jdbc.core.JdbcTemplate

/**
 * @author carl
 * @since 2023-02-17 3:39 AM
 **/
class JdbcFactory(
    jdbcTemplate: JdbcTemplate
) : RepositoryFactory {
    override fun <T, ID> make(): Repository<T, ID> {
        TODO("Not yet implemented")
    }
}