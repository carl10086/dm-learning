package com.ysz.dm.base.repo.support.sql

import com.ysz.dm.base.core.domain.page.PageRequest
import com.ysz.dm.base.core.domain.page.Pageable
import com.ysz.dm.base.core.domain.page.Sort
import com.ysz.dm.base.repo.support.query.OrPart
import com.ysz.dm.base.repo.support.query.Part
import com.ysz.dm.base.repo.support.query.PartTree
import com.ysz.dm.base.repo.support.query.PartType

/**
 * @author carl
 * @since 2023-02-21 2:17 AM
 **/
interface DynamicSqlBuilder {
    fun buildWherePart(ctx: DynamicSqlBuilderContext): String

    fun buildOrderPart(sort: Sort, propertyToColumnMap: Map<String, String>): String

    fun buildPagePart(pageRequest: PageRequest): String

    fun buildLimitPart(max: Int?): String
}

class DynamicSqlBuilderContext(
    val partTree: PartTree,
    val pageRequest: PageRequest?,
    /*a iterator used to iterate args*/
    val argsIterator: ArgsIterator,
    val propertyToColumnMap: Map<String, String>,
    val multiOrPart: Boolean
) {
    companion object {
        fun make(
            pageTree: PartTree,
            args: Array<out Any>?,
            pageAbleArgIndex: Int,
            propertyToColumnMap: Map<String, String>,
        ): DynamicSqlBuilderContext {
            var pureArgs = args
            var pageRequest: PageRequest? = null
            if (pageAbleArgIndex >= 0) {
                pageRequest = args!![pageAbleArgIndex] as PageRequest?
                pureArgs = args.filter { it !is Pageable }.toTypedArray()
            }
            pureArgs = pureArgs ?: emptyArray()
            return DynamicSqlBuilderContext(
                pageTree,
                pageRequest,
                ArgsIterator(pureArgs),
                propertyToColumnMap,
                pageTree.predicate.nodes.size > 1
            )
        }
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

enum class DynamicSqlBuilderImpl : DynamicSqlBuilder {
    MYSQL {
        override fun buildWherePart(ctx: DynamicSqlBuilderContext): String {
            val partTree = ctx.partTree
            val orParts = partTree.predicate.nodes
            return if (orParts.isEmpty()) ""
            else "WHERE " + orParts
                .asSequence()
                .map { orPart2Sql(it, ctx) }
                .joinToString(" or ")
        }


        private fun orPart2Sql(orPart: OrPart, ctx: DynamicSqlBuilderContext): String {
            val children = orPart.children
            val sql = children.asSequence().map { part2Sql(it, ctx) }.joinToString(" and ")


            return if (ctx.multiOrPart && children.size > 1) "($sql)" else sql

        }

        private fun part2Sql(part: Part, ctx: DynamicSqlBuilderContext): String {
            val colName = ctx.propertyToColumnMap[part.propertyPath.name]!!

            val argSize = when (part.type) {
                PartType.IN -> {
                    val argAsCollection = ctx.argsIterator.next() as Collection<Any?>
                    ctx.argsIterator.addAll(argAsCollection)
                    argAsCollection.size
                }

                else -> {
                    for (i in 0 until part.type.numberOfArguments) {
                        ctx.argsIterator.add(ctx.argsIterator.next())
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
    };

    override fun buildOrderPart(sort: Sort, propertyToColumnMap: Map<String, String>): String {
        val orders = sort.orders
        if (orders.isNotEmpty()) {
            return " ORDER BY " + orders.joinToString(",") { "${propertyToColumnMap[it.prop]} ${it.direction}" }
        }

        return ""
    }


    override fun buildPagePart(pageRequest: PageRequest): String {
        return "limit ${pageRequest.offset()}, ${pageRequest.pageSize()}"
    }

    override fun buildLimitPart(max: Int?): String {
        return if (max == null || max <= 0) return "" else " LIMIT $max"
    }


}