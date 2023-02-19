package com.ysz.dm.base.repo.support.query

import com.ysz.dm.base.repo.support.mapping.PropertyPath
import org.springframework.data.util.TypeInformation
import java.beans.Introspector

/**
 * @author carl
 * @since 2023-02-19 3:19 PM
 **/
class Part(
    val type: PartType, val propertyPath: PropertyPath
) {
    companion object {
        fun fromSource(source: String, clazz: Class<*>): Part {
            val type = PartType.fromProperty(source)

            return Part(
                type,
                PropertyPath(type.extractProperty(source), TypeInformation.of(clazz))
            )

        }
    }

}

enum class PartType(
    val keywords: List<String>, val numberOfArguments: Int
) {
    BETWEEN(2, "IsBetween", "Between"),
    IS_NOT_NULL(0, "IsNotNull", "NotNull"),
    IS_NULL(
        0,
        "IsNull",
        "Null"
    ),
    LESS_THAN("IsLessThan", "LessThan"), LESS_THAN_EQUAL(
        "IsLessThanEqual",
        "LessThanEqual"
    ),
    GREATER_THAN("IsGreaterThan", "GreaterThan"), GREATER_THAN_EQUAL(
        "IsGreaterThanEqual",
        "GreaterThanEqual"
    ),
    BEFORE("IsBefore", "Before"), AFTER("IsAfter", "After"), NOT_LIKE("IsNotLike", "NotLike"), LIKE(
        "IsLike",
        "Like"
    ),
    STARTING_WITH("IsStartingWith", "StartingWith", "StartsWith"), ENDING_WITH(
        "IsEndingWith",
        "EndingWith",
        "EndsWith"
    ),
    IS_NOT_EMPTY(0, "IsNotEmpty", "NotEmpty"), IS_EMPTY(0, "IsEmpty", "Empty"), NOT_CONTAINING(
        "IsNotContaining",
        "NotContaining",
        "NotContains"
    ),
    CONTAINING("IsContaining", "Containing", "Contains"), NOT_IN("IsNotIn", "NotIn"),
    IN(
        "IsIn",
        "In"
    ),
    NEAR("IsNear", "Near"), WITHIN("IsWithin", "Within"), REGEX("MatchesRegex", "Matches", "Regex"), EXISTS(
        0,
        "Exists"
    ),
    TRUE(0, "IsTrue", "True"), FALSE(0, "IsFalse", "False"), NEGATING_SIMPLE_PROPERTY(
        "IsNot",
        "Not"
    ),
    SIMPLE_PROPERTY("Is", "Equals");

    companion object {
        val ALL = arrayOf(
            BETWEEN,
            IS_NOT_NULL,
            IS_NULL,
            LESS_THAN,
            LESS_THAN_EQUAL,
            GREATER_THAN,
            GREATER_THAN_EQUAL,
            IN
        )

        val ALL_KEYWORDS = ALL.asSequence().map { it.keywords }.flatten().toList()

        fun fromProperty(rawProperty: String): PartType {
            return ALL.find { it.supports(rawProperty) } ?: SIMPLE_PROPERTY
        }

    }

    constructor(numberOfArguments: Int, vararg keywords: String) : this(
        keywords.toList(), numberOfArguments
    )

    constructor(vararg keywords: String) : this(
        keywords.toList(), 1
    )

    /**
     * callback method to extract the actual propertyPath to be found from the given part .
     * Strips the keyword from the part's end if available
     */
    fun extractProperty(part: String): String {
        val candidate = Introspector.decapitalize(part)

        for (keyword in keywords) {
            if (candidate.endsWith(keyword)) {
                return candidate.substring(0, candidate.length - keyword.length)
            }
        }

        return candidate
    }

    /**
     * returns whether the type supports the given raw property .
     *
     * Default implementation checks whether the property ends with the registered keyword .
     *
     * Does not support the keyword if the property is a valid field as is .
     */
    private fun supports(property: String): Boolean =
        this.keywords.firstOrNull() {
            property.endsWith(it)
        } != null

}