package com.ysz.dm.base.repo.impl.jdbctpl

import com.ysz.dm.base.core.domain.page.Page
import com.ysz.dm.base.core.domain.page.PageImpl
import com.ysz.dm.base.repo.repository.CrudRepository
import com.ysz.dm.base.repo.repository.DomainClassType.Companion.ALL
import com.ysz.dm.base.repo.repository.InvokeMethodMeta
import com.ysz.dm.base.repo.repository.RepositoryMeta
import com.ysz.dm.base.repo.support.mapping.*
import com.ysz.dm.base.repo.support.sql.DynamicSqlBuilderContext
import com.ysz.dm.base.repo.support.sql.DynamicSqlBuilderImpl
import org.slf4j.LoggerFactory
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.jdbc.core.RowMapper
import org.springframework.jdbc.support.GeneratedKeyHolder
import org.springframework.jdbc.support.KeyHolder
import java.sql.PreparedStatement
import kotlin.reflect.KClass


/**
 * @author carl
 * @since 2023-02-17 3:41 AM
 **/

class SimpleJdbcRepository<T : Any, ID>(
    repositoryMeta: RepositoryMeta,
    private val tableName: String,
    converter: PropertyColNameConverter = PropertyColNameConverter.CAMEL_TO_UNDERSCORE,
    private val jdbcTpl: JdbcTemplate,
) : CrudRepository<T, ID> {

    private var entityMapper: ReflectEntityMapper<T>
    private var columns: String
    private val rowMapper: RowMapper<T>
    private val meta: ReflectDomainMeta<T>
    private val sqlBuilder = DynamicSqlBuilderImpl.MYSQL


    init {
        check(ALL.contains(repositoryMeta.domainClassType)) {
            "currently not support for this type"
        }

        val domainTypeKClass = repositoryMeta.domainKClass as KClass<T>

        this.entityMapper = KotlinDataClassMapper(
            domainTypeKClass,
            converter,
            PropertyValueMapperDefaultImpl
        )

        this.meta = this.entityMapper.domainMeta()
        this.columns = meta.columns.joinToString(",")
        this.rowMapper = RowMapper { rs, _ ->
            this.entityMapper.rowMapper()(rs)
        }


    }

    override fun findById(id: ID): T? {
        val sql = """
            SELECT $columns FROM $tableName WHERE ${meta.primaryKeyColumn} = ?
        """.trimIndent()


        return this.jdbcTpl.query(
            sql,
            this.rowMapper,
            id
        ).firstOrNull()
    }

    override fun insert(entity: T): T {
        /*1. if support auto generate id and also entity id is null*/
        if (meta.autoGenerateId && entityMapper.primaryKeyValue(entity) == null) {
            val keyHolder: KeyHolder = GeneratedKeyHolder()
            val columnsWithoutPk = meta.columnsWithoutPrimaryKey();
            this.jdbcTpl.update(
                {
                    val sql = """
                        INSERT INTO $tableName (${columnsWithoutPk.joinToString(",")}) 
                        VALUES
                        (${columnsWithoutPk.joinToString(",") { "?" }})
                        """.trimIndent()
                    val stat = it.prepareStatement(sql, PreparedStatement.RETURN_GENERATED_KEYS)
                    var count = 1
                    for (value in entityMapper.propertyValues(entity, false)) {
                        /*maybe need type mapping, may jdbc already handle this for you , haha*/
                        stat.setObject(count++, value)
                    }
                    stat
                }, keyHolder
            )
            keyHolder.key?.let { this.entityMapper.setPrimaryKeyValue(entity, it) }
        } else {
            this.jdbcTpl.update("""
                INSERT INTO $tableName ($columns) VALUES(${
                meta.columns.joinToString(",") { "?" }
            })
                """.trimIndent(), *entityMapper.propertyValues(entity).toTypedArray())
        }

        return entity
    }

    override fun queryByIds(ids: List<ID>): List<T> {
        val sql = """
             SELECT $columns FROM $tableName WHERE ${meta.primaryKeyColumn} in (${ids.joinToString(",") { "?" }})
            """.trimIndent()

        val array: Array<Any> = Array(ids.size) {
            ids[it] as Any
        }

        return this.jdbcTpl.query(sql, rowMapper, *array)
    }

    override fun update(entity: T) {
        val primaryKeyColumnName = meta.primaryKeyColumn
        val sql = """
            UPDATE $tableName SET ${
            meta.columns.filter { it != primaryKeyColumnName }.joinToString(",") { "${it}=?" }
        } 
            WHERE $primaryKeyColumnName = ?
        """.trimIndent()

        val params = buildList {
            addAll(entityMapper.propertyValues(entity, false))
            add(entityMapper.primaryKeyValue(entity))
        }

        this.jdbcTpl.update(sql, *params.toTypedArray())
    }


    fun doInvoke(funcMeta: InvokeMethodMeta, args: Array<out Any>?): Any? {
        val ctx = makeDynamicSqlBuilderContext(funcMeta, args)
        val wherePart = sqlBuilder.buildWherePart(ctx)
        val subject = funcMeta.partTree.subject

        return when {
            funcMeta.pageAbleArgIndex >= 0 -> pageQuery(ctx, wherePart)
            subject.count -> countQuery(ctx, wherePart, funcMeta)
            subject.exists -> countQuery(ctx, wherePart, funcMeta).toLong() > 0L
            else -> normalQuery(ctx, wherePart, funcMeta)
        }
    }

    private fun countQuery(ctx: DynamicSqlBuilderContext, wherePart: String, funcMeta: InvokeMethodMeta): Number {
        val countSql = """
                SELECT COUNT(*) as total FROM $tableName $wherePart
            """.trimIndent()

        val total = jdbcTpl.queryForObject(countSql, Long::class.java, *ctx.argsIterator.list.toTypedArray())

        return if (funcMeta.resultType.isInt) total.toInt() else total
    }

    private fun normalQuery(ctx: DynamicSqlBuilderContext, wherePart: String, funcMeta: InvokeMethodMeta): Any? {
        val sql = """
                SELECT $columns FROM $tableName $wherePart
            """.trimIndent()

        val result = this.jdbcTpl.query(
            sql,
            this.rowMapper,
            *ctx.argsIterator.list.toTypedArray()
        )

        return if (funcMeta.resultType.isCollection) result else result.firstOrNull()
    }

    private fun pageQuery(ctx: DynamicSqlBuilderContext, wherePart: String): Any? {
        val countSql = """
                SELECT COUNT(*) as total FROM $tableName $wherePart
            """.trimIndent()

        val total = jdbcTpl.queryForObject(countSql, Long::class.java, *ctx.argsIterator.list.toTypedArray())
        if (total == 0L) {
            return Page.empty<T>()
        }

        val pageRequest = ctx.pageRequest!!

        val querySql = """
                 SELECT  ${columns} FROM $tableName 
                 $wherePart 
                 ${sqlBuilder.buildOrderPart(pageRequest.sort, meta.propertyToColumnMap)} 
                 ${sqlBuilder.buildPagePart(pageRequest)}
            """.trimIndent()


        val contentList = this.jdbcTpl.query(querySql, this.rowMapper, *ctx.argsIterator.list.toTypedArray())

        return PageImpl(contentList, total, pageRequest)
    }

    private fun makeDynamicSqlBuilderContext(
        funcMeta: InvokeMethodMeta,
        args: Array<out Any>?
    ): DynamicSqlBuilderContext {
        return DynamicSqlBuilderContext.make(
            funcMeta.partTree,
            args,
            funcMeta.pageAbleArgIndex,
            meta.propertyToColumnMap
        )
    }


    companion object {
        private val logger = LoggerFactory.getLogger(SimpleJdbcRepository::class.java)
    }

}


