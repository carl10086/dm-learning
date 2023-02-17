package com.ysz.dm.base.repo.support

import java.sql.ResultSet

/**
 * @author carl
 * @since 2023-02-17 10:17 PM
 **/
interface ResultSetRowMapper<T> {
    fun map(rs: ResultSet)
}