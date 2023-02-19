package com.ysz.dm.base.repo.impl.jdbctpl

import com.ysz.dm.base.repo.common.Student
import com.ysz.dm.base.repo.common.StudentRepo
import com.ysz.dm.base.repo.common.StudentSchemaTools
import com.ysz.dm.base.repo.repository.RepositoryMeta
import com.ysz.dm.base.test.eq
import org.junit.jupiter.api.BeforeAll
import org.junit.jupiter.api.Test

/**
 * @author carl
 * @since 2023-02-17 10:29 PM
 */
internal class SimpleJdbcRepositoryTest {

    @Test
    fun `test findById`() {
        jdbcRepo.findById(2L)!!.username.eq("user2")
    }

    @Test
    fun `test update`() {
        jdbcRepo.update(jdbcRepo.findById(1L)!!.apply {
            this.username = "abc"
        }
        )

        jdbcRepo.findById(1L)!!.username eq "abc"
    }


    companion object {


        @BeforeAll
        @JvmStatic
        fun setUp() {
            StudentSchemaTools.init()
        }

        val jdbcRepo = SimpleJdbcRepository<Student, Long>(
            repositoryMeta = RepositoryMeta(StudentRepo::class.java),
            tableName = "t_students",
            jdbcTpl = StudentSchemaTools.jdbcTpl
        )
    }
}