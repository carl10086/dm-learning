package com.ysz.dm.base.repo.support.query

import com.ysz.dm.base.core.domain.page.Direction
import com.ysz.dm.base.core.domain.page.Order
import java.beans.Introspector
import java.util.regex.Pattern

/**
 * @author carl
 * @since 2023-02-19 2:24 AM
 **/
class PartTree(
    source: String
) {

    val subject: Subject
    val predicate: Predicate

    init {
        val matcher = PREFIX_TEMPLATE.matcher(source)

        if (matcher.find()) {
            this.subject = Subject(matcher.group(0))
            this.predicate = Predicate.fromSource(
                source.substring(matcher.group().length)
            )
        } else {

            this.subject = Subject("")
            this.predicate = Predicate.fromSource(source)
        }


    }


    companion object {
        const val KEYWORD_TEMPLATE = "(%s)(?=(\\p{Lu}|\\P{InBASIC_LATIN}))"

        /*this patterns mean normal search and return result domain list*/
        const val QUERY_PATTERN = "find|read|get|query|search|page|stream"

        /*this means count search and return count*/
        const val COUNT_PATTERN = "count"

        /*this mean return true or false*/
        const val EXISTS_PATTERN = "exists"

        /*this means just delete or remove*/
        const val DELETE_PATTERN = "delete|remove"
        val PREFIX_TEMPLATE = Pattern.compile( //
            "^($QUERY_PATTERN|$COUNT_PATTERN|$EXISTS_PATTERN|$DELETE_PATTERN)((\\p{Lu}.*?))??By"
        )

        fun split(text: String, keyword: String): Array<String> {
            return Pattern.compile(String.format(KEYWORD_TEMPLATE, keyword)).split(text)
        }


    }
}

/**
 *  封装了一个 sql 语句的整体特征. 比如说类型是i count or exists or delete or distinct
 */
data class Subject(
    val subject: String
) {
    val distinct: Boolean = if (subject.isNotBlank()) subject.contains(DISTINCT) else false
    val count = if (subject.isNotBlank()) COUNT_BY_TEMPLATE.matcher(subject).find() else false
    val exists = if (subject.isNotBlank()) EXISTS_BY_TEMPLATE.matcher(subject).find() else false
    val delete = if (subject.isNotBlank()) DELETE_BY_TEMPLATE.matcher(subject).find() else false
    val maxResults: Int? = if (subject.isNotBlank()) returnMaxResultsIfFirstKSubjectOrNull() else null


    private fun returnMaxResultsIfFirstKSubjectOrNull(): Int? {
        val grp = LIMITED_QUERY_TEMPLATE.matcher(this.subject)
        if (!grp.find()) return null

        val group4 = grp.group(4)
        return if (group4.isNullOrBlank()) {
            1
        } else
            group4.toInt()
    }

    companion object {
        private val DISTINCT = "Distinct"
        private val COUNT_BY_TEMPLATE = Pattern.compile("^count(\\p{Lu}.*?)??By")
        private val EXISTS_BY_TEMPLATE = Pattern.compile("^(" + PartTree.EXISTS_PATTERN + ")(\\p{Lu}.*?)??By")
        private val DELETE_BY_TEMPLATE = Pattern.compile("^(" + PartTree.DELETE_PATTERN + ")(\\p{Lu}.*?)??By")
        private val LIMITING_QUERY_PATTERN = "(First|Top)(\\d*)?"
        private val LIMITED_QUERY_TEMPLATE =
            Pattern.compile("^(" + PartTree.QUERY_PATTERN + ")(" + DISTINCT + ")?" + LIMITING_QUERY_PATTERN + "(\\p{Lu}.*?)??By")


    }
}

data class OrPart(
    val children: List<Part>
) {
    constructor(source: String) : this(
        PartTree.split(source, "And")
            .filter(String::isNotBlank)
            .map { Part.fromSource(it) }
            .toList()
    )

    override fun toString(): String {
        return this.children.joinToString(" and ")
    }
}

data class Predicate(
    val nodes: List<OrPart>,
    val orderBySource: OrderBySource
) {
    companion object {

        private const val ORDER_BY = "OrderBy"

        fun fromSource(
            predicate: String,
        ): Predicate {

            val parts = PartTree.split(predicate, ORDER_BY)
            return Predicate(
                PartTree.split(parts[0], "Or")
                    .filter { it.isNotBlank() }
                    .map { OrPart(it) }
                    .toList(),

                if (parts.size == 2) OrderBySource.fromClause(parts[1]) else OrderBySource.EMPTY
            )
        }
    }

}


data class OrderBySource(val orders: List<Order>) {
    companion object {

        val EMPTY = OrderBySource(emptyList())
        private const val BLOCK_SPLIT = "(?<=Asc|Desc)(?=\\p{Lu})"
        private val DIRECTION_SPLIT = Pattern.compile("(.+?)(Asc|Desc)?$")
        private val DIRECTION_KEYWORDS: Set<String> = HashSet(mutableListOf("Asc", "Desc"))

        fun fromClause(clause: String): OrderBySource {
            val orderList = ArrayList<Order>()

            if (clause.isNotBlank()) {
                for (part in PartTree.split(clause, BLOCK_SPLIT)) {
                    val matcher = DIRECTION_SPLIT.matcher(part)

                    if (!matcher.find()) throw IllegalArgumentException("Invalid order syntax for part $part")

                    val propertyString = matcher.group(1)
                    val directionString = matcher.group(2)
                    val direction = when {
                        directionString == null -> Direction.DESC
                        directionString.lowercase() == "asc" -> Direction.ASC
                        else -> Direction.DESC
                    }

                    if (DIRECTION_KEYWORDS.contains(propertyString) && directionString == null) throw IllegalArgumentException(
                        "Invalid order syntax for part $part"
                    )

                    orderList.add(
                        Order(
                            direction,
                            Introspector.decapitalize(propertyString)
                        )
                    )

                }
            }
            return OrderBySource(orderList.toList())
        }

    }
}