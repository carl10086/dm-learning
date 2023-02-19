package com.ysz.dm.base.repo.support.mapping

import com.google.common.base.CaseFormat

/**
 * @author carl
 * @since 2023-02-17 9:06 PM
 **/

enum class PropertyColNameConverter(
    private val propertyToColFunc: (String) -> String,
    private val colToPropertyFunc: (String) -> String
) {
    ORIGIN(
        { it },
        { it }
    ),
    CAMEL_TO_UNDERSCORE(
        { CaseFormat.LOWER_CAMEL.to(CaseFormat.LOWER_UNDERSCORE, it) },
        { CaseFormat.LOWER_UNDERSCORE.to(CaseFormat.LOWER_CAMEL, it) }
    );

    fun propertyToColName(propertyName: String): String = propertyToColFunc(propertyName)
    fun colToPropertyName(colName: String): String = colToPropertyFunc(colName)

}




