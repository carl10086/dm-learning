package com.ysz.dm.base.repo.repository

import org.springframework.data.util.TypeInformation
import kotlin.reflect.KClass
import kotlin.reflect.KMutableProperty1
import kotlin.reflect.KType
import kotlin.reflect.KVisibility
import kotlin.reflect.full.isSuperclassOf
import kotlin.reflect.full.memberProperties
import kotlin.reflect.jvm.jvmErasure

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
class RepositoryMetaV1(repositoryInterface: Class<*>) {
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

class RepositoryMeta(
    val domainKClass: KClass<*>, val idKClass: KClass<*>, val domainClassType: DomainClassType
) {
    companion object {
        fun fromRepoInf(inf: KClass<*>): RepositoryMeta {
            check(inf.java.isInterface) { "$inf must  be a interface class" }
            val type = findRepositoryType(inf)
            checkNotNull(type) { "$inf must impl Repository interface" }

            val domainClass = type.arguments[0].type!!.jvmErasure

            return RepositoryMeta(
                domainClass,
                type.arguments[1].type!!.jvmErasure,
                DomainClassType.fromKClass(domainClass)
            )
        }


        fun findRepositoryType(inf: KClass<*>): KType? {
            for (supertype in inf.supertypes) {
                if (Repository::class.isSuperclassOf(supertype.jvmErasure)) {
                    return supertype
                }
            }

            return null
        }
    }


}


enum class DomainClassType {
    /**
     * suitAble to kotlin data class
     */
    KOTLIN_DATA_CLASS,

    /**
     * java 17 record model
     */
    JAVA_RECORD,

    /**
     * classic java bean model. with No-Args constructor and get .setter?
     */
    JAVA_BEAN;

    companion object {
        fun fromKClass(kClass: KClass<*>): DomainClassType {
            return when {
                kClass.isData -> KOTLIN_DATA_CLASS
                kClass.java.isRecord -> JAVA_RECORD
                checkJavaBeanForKClass(kClass) -> JAVA_BEAN
                else -> throw IllegalArgumentException("no support for kClass ${kClass}")
            }
        }

        fun checkJavaBeanForKClass(kClass: KClass<*>): Boolean {
            if (!kClass.constructors.any { it.parameters.isEmpty() }) {
                return false
            }
            kClass.memberProperties.forEach {
                if (it.getter.visibility != KVisibility.PUBLIC) return false
                if (it !is KMutableProperty1<out Any, *> || it.setter.visibility != KVisibility.PUBLIC) return false
            }

            return true

        }
    }
}