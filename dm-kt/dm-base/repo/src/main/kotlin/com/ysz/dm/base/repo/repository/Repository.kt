package com.ysz.dm.base.repo.repository

import org.springframework.data.util.TypeInformation
import java.lang.reflect.Method

/**
 * Central repository marker inf .
 *
 *
 * @param T the domain type of repository manages
 * @param ID the type of id of the entity the repository manage
 * @author carl
 * @since 2023-02-17 12:15 AM
 **/
interface Repository<T, ID> {}


/**
 * Metadata of repository interface
 */
class RepositoryMeta(repositoryInterface: Class<*>) {
    val domainType: TypeInformation<*>
    val idType: TypeInformation<*>

    init {
        val arguments = TypeInformation.of(repositoryInterface)
            .getRequiredSuperTypeInformation(Repository::class.java).typeArguments
        check(arguments.size >= 2) { "could not resolve domain type and id type" }
        domainType = arguments[0]!!
        idType = arguments[1]!!
    }
}