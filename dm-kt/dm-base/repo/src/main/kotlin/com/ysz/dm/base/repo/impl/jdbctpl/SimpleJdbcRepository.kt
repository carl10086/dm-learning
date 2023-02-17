package com.ysz.dm.base.repo.impl.jdbctpl

import com.ysz.dm.base.repo.anotation.Ignore
import com.ysz.dm.base.repo.repository.CrudRepository
import com.ysz.dm.base.repo.repository.RepositoryMeta
import com.ysz.dm.base.repo.support.PropertyColNameConverter
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.jdbc.core.RowMapper
import java.lang.reflect.Field
import kotlin.reflect.full.memberProperties
import kotlin.reflect.jvm.javaField

/**
 * @author carl
 * @since 2023-02-17 3:41 AM
 **/

class SimpleJdbcRepository<T, ID>(
    private val repositoryMeta: RepositoryMeta,
    private val tableName: String,
    private val converter: PropertyColNameConverter = PropertyColNameConverter.CAMEL_TO_UNDERSCORE,
    private val jdbcTpl: JdbcTemplate,
) : CrudRepository<T, ID> {

    private val fields: List<Field>

    private val cols: String

    private val colNameSet: Set<String>

    private val rowMapper: RowMapper<T>

    init {
        val domainType = repositoryMeta.domainType.type
        val domainTypeKClass = domainType.kotlin
        val isDataClass = domainTypeKClass.isData
        check(isDataClass)

        /*we need to use constructor for kotlin data class*/
        val properties = domainTypeKClass.memberProperties.filter {
            it.javaField!!.getAnnotation(Ignore::class.java) == null
        }

        this.fields = properties.map { it.javaField!! }

        this.colNameSet = properties.map { it.name }.toSortedSet()
        /*find a constructors include all*/
        val constructor = domainTypeKClass.constructors.firstOrNull { constructor ->
            val params = constructor.parameters
            params.map { it.name }.containsAll(colNameSet)
        }

        check(constructor != null) { "we can't find a constructor for data class" }
        this.cols = colNameSet.joinToString(",")




        this.rowMapper = RowMapper { rs, _ ->

            val params = constructor.parameters
            val list = buildList {
                for (param in params) {
                    if (colNameSet.contains(param.name)) {
                        add(rs.getObject(param.name))
                    } else {
                        add(null)
                    }
                }
            }

            @Suppress("UNCHECKED_CAST")
            constructor.call(*list.toTypedArray()) as T
        }
    }

    override fun findById(id: ID): T? {
        val sql = """
            SELECT $cols FROM $tableName WHERE id = ?
        """.trimIndent()


        return this.jdbcTpl.query(
            sql,
            this.rowMapper,
            id
        ).firstOrNull()
    }

    override fun insertOne(entity: T) {
        val sql = """
            INSERT INTO $tableName (${this.cols}) VALUES(${
            this.colNameSet.joinToString(",") { "?" }
        })
            """.trimIndent()


        val params = fields.map { it.apply { trySetAccessible() }.get(entity) }

        this.jdbcTpl.update(sql, *params.toTypedArray())
    }

}
