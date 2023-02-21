package com.ysz.dm.base.repo.support.query

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
        const val QUERY_PATTERN = "find|read|get|query|search|stream"

        /*this means count search and return count*/
        const val COUNT_PATTERN = "count"

        /*this mean return true or false*/
        const val EXISTS_PATTERN = "exists"

        /*this means just delete or remove*/
        const val DELETE_PATTERN = "delete|remove"
        val PREFIX_TEMPLATE = Pattern.compile( //
            "^($QUERY_PATTERN|$COUNT_PATTERN|$EXISTS_PATTERN|$DELETE_PATTERN)((\\p{Lu}.*?))??By"
        )

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
        source
            .splitToSequence("And")
            .filter(String::isNotBlank)
            .map { Part.fromSource(it) }
            .toList()
    )

    override fun toString(): String {
        return this.children.joinToString(" and ")
    }
}

data class Predicate(
    val nodes: List<OrPart>
) {
    companion object {
        fun fromSource(
            source: String,
        ): Predicate = Predicate(
            source
                .splitToSequence("Or")
                .filter { it.isNotBlank() }
                .map { OrPart(it) }
                .toList()
        )
    }

}