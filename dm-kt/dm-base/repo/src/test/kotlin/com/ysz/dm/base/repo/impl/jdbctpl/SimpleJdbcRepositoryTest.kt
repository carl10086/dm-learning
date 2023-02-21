package com.ysz.dm.base.repo.impl.jdbctpl

import com.ysz.dm.base.repo.common.Student
import com.ysz.dm.base.repo.common.StudentRepo
import com.ysz.dm.base.repo.common.StudentSchemaTools
import com.ysz.dm.base.repo.repository.Repository
import com.ysz.dm.base.repo.repository.RepositoryMeta
import com.ysz.dm.base.test.eq
import org.junit.jupiter.api.BeforeAll
import org.junit.jupiter.api.Test
import org.springframework.data.util.TypeInformation

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


    @Test
    fun `test type`() {
        val type = TypeInformation.of(StudentRepo::class.java).getRequiredSuperTypeInformation(
            Repository::class.java
        )
        val args = type.typeArguments
        println(type)
    }

    companion object {


        @BeforeAll
        @JvmStatic
        fun setUp() {
            StudentSchemaTools.init()
        }

        val jdbcRepo = SimpleJdbcRepository<Student, Long>(
            repositoryMeta = RepositoryMeta.fromRepoInf(StudentRepo::class),
            tableName = "t_students",
            jdbcTpl = StudentSchemaTools.jdbcTpl
        )
    }
}