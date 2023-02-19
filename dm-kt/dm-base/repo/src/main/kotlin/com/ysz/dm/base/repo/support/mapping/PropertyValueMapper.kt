package com.ysz.dm.base.repo.support.mapping

import java.math.BigDecimal
import java.sql.ResultSet
import kotlin.reflect.KClass

/**
 * map a jdbc field to a java property
 * @author carl
 * @since 2023-02-18 9:41 PM
 **/
interface PropertyValueMapper {
    fun getValueFromResultSet(
        rs: ResultSet,
        columnName: String,
        requiredType: KClass<*>,
    ): Any?
}



object PropertyValueMapperDefaultImpl : PropertyValueMapper {
    override fun getValueFromResultSet(rs: ResultSet, columnName: String, requiredType: KClass<*>): Any? {
        return when (requiredType) {
            String::class -> rs.getString(columnName)
            Boolean::class -> rs.getBoolean(columnName)
            Byte::class -> rs.getByte(columnName)
            Short::class -> rs.getShort(columnName)
            Int::class -> rs.getInt(columnName)
            Long::class -> rs.getLong(columnName)
            Float::class -> rs.getFloat(columnName)
            BigDecimal::class -> rs.getBigDecimal(columnName)
            java.util.Date::class -> rs.getTimestamp(columnName)?.let { java.util.Date(it.time) }
            else -> throw IllegalArgumentException("requiredType $columnName not supported")
        }
    }


}
