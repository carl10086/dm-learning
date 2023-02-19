package com.ysz.dm.base.repo.impl.jdbctpl

import com.ysz.dm.base.repo.repository.CrudRepository
import com.ysz.dm.base.repo.repository.RepositoryMeta
import com.ysz.dm.base.repo.support.mapping.KotlinDataClassMapper
import com.ysz.dm.base.repo.support.mapping.PropertyColNameConverter
import com.ysz.dm.base.repo.support.mapping.ReflectEntityMapper
import com.ysz.dm.base.repo.support.query.OrPart
import com.ysz.dm.base.repo.support.query.Part
import com.ysz.dm.base.repo.support.query.PartTree
import com.ysz.dm.base.repo.support.query.PartType
import org.slf4j.LoggerFactory
import org.springframework.data.util.TypeInformation
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.jdbc.core.RowMapper
import org.springframework.jdbc.support.GeneratedKeyHolder
import org.springframework.jdbc.support.KeyHolder
import java.sql.PreparedStatement
import kotlin.reflect.KClass
import kotlin.reflect.KFunction
import kotlin.reflect.jvm.jvmErasure


/**
 * @author carl
 * @since 2023-02-17 3:41 AM
 **/

class SimpleJdbcRepository<T : Any, ID>(
    private val repositoryMeta: RepositoryMeta,
    private val tableName: String,
    private val converter: PropertyColNameConverter = PropertyColNameConverter.CAMEL_TO_UNDERSCORE,
    private val jdbcTpl: JdbcTemplate,
) : CrudRepository<T, ID> {

    private var entityMapper: ReflectEntityMapper<T>
    private var columnsJoinString: String
    private var columns: List<String>
    private val rowMapper: RowMapper<T>


    init {
        val domainTypeKClass = repositoryMeta.domainType.type.kotlin as KClass<T>

        this.entityMapper = KotlinDataClassMapper(
            domainTypeKClass,
            converter
        )

        this.columns = this.entityMapper.columns()
        this.columnsJoinString = this.columns.joinToString(",")
        this.rowMapper = RowMapper { rs, _ ->
            this.entityMapper.rowMapper()(rs)
        }


    }

    override fun findById(id: ID): T? {
        val sql = """
            SELECT $columnsJoinString FROM $tableName WHERE id = ?
        """.trimIndent()


        return this.jdbcTpl.query(
            sql,
            this.rowMapper,
            id
        ).firstOrNull()
    }

    override fun insert(entity: T) {

        val autoGenerateId = entityMapper.autoGenerateId()
        if (autoGenerateId && entityMapper.primaryKeyValue(entity) == null) {
            val keyHolder: KeyHolder = GeneratedKeyHolder()
            val columnsWithoutPk = columns.filter { it != entityMapper.primaryKeyColumnName() }
            this.jdbcTpl.update(
                {
                    val sql = """
                        INSERT INTO $tableName (${columnsWithoutPk.joinToString(",")}) VALUES(${
                        columnsWithoutPk.joinToString(",") { "?" }
                    })
                        """.trimIndent()
                    val stat = it.prepareStatement(sql, PreparedStatement.RETURN_GENERATED_KEYS)
                    var count = 1
                    for (allPropertyValue in entityMapper.allPropertyValues(entity, false)) {
                        stat.setObject(count++, allPropertyValue)
                    }
                    stat
                }, keyHolder
            )
            keyHolder.key?.let { this.entityMapper.setPrimaryKeyValue(entity, it) }
        } else {
            val params = this.entityMapper.allPropertyValues(entity)
            this.jdbcTpl.update("""
                INSERT INTO $tableName ($columnsJoinString) VALUES(${
                columns.joinToString(",") { "?" }
            })
                """.trimIndent(), *params.toTypedArray())
        }


    }

    override fun queryByIds(ids: List<ID>): List<T> {
        val sql = """
             SELECT $columnsJoinString FROM $tableName WHERE id in (${ids.joinToString(",") { "?" }})
            """.trimIndent()

        val array: Array<Any> = Array(ids.size) {
            ids[it] as Any
        }

        return this.jdbcTpl.query(sql, rowMapper, *array)
    }

    override fun update(entity: T) {
        val primaryKeyColumnName = entityMapper.primaryKeyColumnName()
        val sql = """
            UPDATE $tableName SET ${
            columns.filter { it != primaryKeyColumnName }.joinToString(",") { "${it}=?" }
        } 
            WHERE ${entityMapper.primaryKeyColumnName()} = ?
        """.trimIndent()

        val params = buildList {
            addAll(entityMapper.allPropertyValues(entity, false))
            add(entityMapper.primaryKeyValue(entity))
        }

        this.jdbcTpl.update(sql, *params.toTypedArray())
    }

    fun doInvoke(func: KFunction<*>, args: Array<out Any>?): Any? {
        val name = func.name
        val domainClazz = this.repositoryMeta.domainType.type
        val partTree = PartTree(name, domainClazz)
        val orParts = partTree.predicate.nodes
        val subject = partTree.subject


        val needParenthesis = orParts.size > 1

        val argsIterator = ArgsIterator(args)
        val whereCause = orParts
            .asSequence()
            .map { orPart2Sql(it, needParenthesis, argsIterator) }
            .joinToString(" or ")


        val sql = """
                SELECT $columnsJoinString FROM $tableName WHERE $whereCause
            """.trimIndent()

        val result = if (null == args) {
            this.jdbcTpl.query(
                sql,
                this.rowMapper,
            )
        } else {
            if (logger.isDebugEnabled) {
                logger.debug("invoke parameters:${argsIterator.list}")
            }

            this.jdbcTpl.query(
                sql,
                this.rowMapper,
                *argsIterator.list.toTypedArray()
            )

        }
        val collectionLike = TypeInformation.of(func.returnType.jvmErasure.java).isCollectionLike

        return if (collectionLike) result else result.firstOrNull()
    }

    private fun orPart2Sql(orPart: OrPart, needParenthesis: Boolean, argsIterator: ArgsIterator): String {
        val children = orPart.children
        val sql = children.asSequence().map { part2Sql(it, argsIterator) }.joinToString(" and ")


        return if (needParenthesis && children.size > 1) "($sql)" else sql

    }

    private fun part2Sql(part: Part, argsIterator: ArgsIterator): String {
        val prop = part.propertyPath.name
        val colName = this.converter.propertyToColName(prop)

        val argSize = when (part.type) {
            PartType.IN -> {
                val argAsCollection = argsIterator.next() as Collection<Any?>
                argsIterator.addAll(argAsCollection)
                argAsCollection.size
            }

            else -> {
                for (i in 0 until part.type.numberOfArguments) {
                    argsIterator.add(argsIterator.next())
                }
                1
            }
        }

        return when (part.type) {
            PartType.BETWEEN -> "$colName between ? and ?"
            PartType.IS_NOT_NULL -> "$colName is not null"
            PartType.IS_NULL -> "$colName is null"
            PartType.LESS_THAN -> "$colName < ?"
            PartType.LESS_THAN_EQUAL -> "$colName <= ?"
            PartType.GREATER_THAN -> "$colName > ?"
            PartType.GREATER_THAN_EQUAL -> "$colName >= ?"
            PartType.SIMPLE_PROPERTY -> "$colName = ?"
            PartType.IN -> "$colName in (${(0 until argSize).joinToString(",") { "?" }})"
            else -> throw IllegalArgumentException("currently not support ${part.type}")
        }

    }

    companion object {
        private val logger = LoggerFactory.getLogger(SimpleJdbcRepository::class.java)
    }

}


class ArgsIterator(
    private val args: Array<out Any>?
) {
    private var count = 0
    var list: ArrayList<Any?> = ArrayList(args?.size ?: 0)

    fun next(): Any? = this.args?.get(count++)
    fun add(arg: Any?) = this.list.add(arg)

    fun addAll(args: Collection<Any?>) =
        this.list.addAll(args)

}
