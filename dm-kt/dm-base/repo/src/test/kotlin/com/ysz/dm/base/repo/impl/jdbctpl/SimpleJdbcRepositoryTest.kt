package com.ysz.dm.base.repo.impl.jdbctpl

import com.ysz.dm.base.repo.common.Student
import com.ysz.dm.base.repo.common.StudentRepo
import com.ysz.dm.base.repo.repository.RepositoryMeta
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.jdbc.datasource.DriverManagerDataSource
import javax.sql.DataSource

/**
 * @author carl
 * @since 2023-02-17 10:29 PM
 */
internal class SimpleJdbcRepositoryTest {
    val ds: DataSource =
        DriverManagerDataSource().apply {
            this.setDriverClassName("org.h2.Driver")
            this.url = "jdbc:h2:mem:myDb;DB_CLOSE_DELAY=-1"
        }

    val jdbcTpl = JdbcTemplate(ds)

    @BeforeEach
    fun setUp() {
        jdbcTpl.execute(
            """
            create table t_students
            (
            id int primary key not null,
            username varchar(255) null
            )
        """.trimIndent()
        )

    }

    private val jdbcRepo = SimpleJdbcRepository<Student, Long>(
        repositoryMeta = RepositoryMeta(StudentRepo::class.java),
        tableName = "t_students",
        jdbcTpl = this.jdbcTpl
    )


    @Test
    fun `test create`() {
        this.jdbcRepo.insertOne(
            Student(100L, "carl")
        )
        println(this.jdbcRepo.findById(100L))
    }
}