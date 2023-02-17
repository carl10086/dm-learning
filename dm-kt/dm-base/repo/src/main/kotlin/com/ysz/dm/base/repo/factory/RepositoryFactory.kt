package com.ysz.dm.base.repo.factory

import com.ysz.dm.base.repo.repository.Repository

/**
 * The repo instance factory
 * @author carl
 * @since 2023-02-17 3:30 AM
 **/
interface RepositoryFactory {

    fun <T, ID> make(): Repository<T, ID>
}