package com.ysz.dm.base.repo.repository

/**
 * interface for general CRUD operations on a repository for a specific type .
 * @author carl
 * @since 2023-02-17 12:17 AM
 **/
interface CrudRepository<T, ID> : Repository<T, ID> {
    fun findById(id: ID): T?

    fun queryByIds(id: List<ID>): List<T>

    fun insert(entity: T);

    fun update(entity: T)
}