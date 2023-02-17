package com.ysz.dm.base.repo.common

import com.ysz.dm.base.repo.anotation.Entity
import com.ysz.dm.base.repo.anotation.Ignore
import com.ysz.dm.base.repo.repository.CrudRepository

/**
 * @author carl
 * @since 2023-02-17 3:39 PM
 **/
@Entity
data class Student(
    val id: Long,
    val username: String,
    @field:Ignore
    val ignore: String? = null
)


interface StudentRepo : CrudRepository<Student, Long>