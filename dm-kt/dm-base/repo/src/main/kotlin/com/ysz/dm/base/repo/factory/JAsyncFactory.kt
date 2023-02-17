package com.ysz.dm.base.repo.factory

import com.ysz.dm.base.repo.repository.Repository

/**
 * @author carl
 * @since 2023-02-17 3:33 AM
 **/
class JAsyncFactory : RepositoryFactory {
    override fun <T, ID> make(): Repository<T, ID> {
        TODO("Not yet implemented")
    }
}