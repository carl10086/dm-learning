package com.ysz.dm.base.repo.common

import com.ysz.dm.base.core.domain.page.PageImpl
import com.ysz.dm.base.core.domain.page.PageRequest
import com.ysz.dm.base.repo.anotation.Entity
import com.ysz.dm.base.repo.anotation.GeneratedValue
import com.ysz.dm.base.repo.anotation.Id
import com.ysz.dm.base.repo.anotation.Ignore
import com.ysz.dm.base.repo.impl.jdbctpl.SimpleJdbcRepositoryTest
import com.ysz.dm.base.repo.repository.CrudRepository
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.jdbc.datasource.DriverManagerDataSource
import java.time.Duration
import java.util.*
import javax.sql.DataSource

/**
 * @author carl
 * @since 2023-02-17 3:39 PM
 **/
@Entity
data class Student(
    @Id
    @GeneratedValue
    var id: Long? = null,
    var username: String,
    @field:Ignore
    val ignore: String? = null,
    val createAt: Date = Date(),
    val age: Int? = 10,
    @field:Ignore
    val gender: Int = 1
)


interface StudentRepo : CrudRepository<Student, Long> {
    fun findByUsername(username: String): Student?
    fun queryByAgeBetween(startAge: Int, endAge: Int): List<Student>
    fun queryByAgeBetweenAndUsernameOrCreateAtGreaterThanEqual(
        startAge: Int, endAge: Int, username: String, startCreateAt: Date
    ): List<Student>

    fun queryByAgeInAndUsername(
        ages: List<Int>,
        username: String
    ): List<Student>

    fun queryByAgeBetween(startAge: Int, endAge: Int, page: PageRequest): PageImpl<Student>
}

object StudentSchemaTools {
    fun beforeHours(hours: Int): Date =
        Date(System.currentTimeMillis() / Duration.ofHours(1L).toMillis() - Duration.ofHours(hours.toLong()).toMillis())

    val ds: DataSource =
        DriverManagerDataSource().apply {
            this.setDriverClassName("org.h2.Driver")
            this.url = "jdbc:h2:mem:myDb;DB_CLOSE_DELAY=-1"
        }

    val jdbcTpl = JdbcTemplate(ds)

    fun init() {
        jdbcTpl.execute(
            """
            create table t_students
            (
            id int  auto_increment primary key not null,
            username varchar(255) not null ,
            create_at timestamp not null ,
            age int not null
            )
        """.trimIndent()
        )
        for (i in 1 until 10) {
            Student(
                username = "user$i",
                age = 10 + i,
                createAt = beforeHours(i)
            ).apply { SimpleJdbcRepositoryTest.jdbcRepo.insert(this) }
        }

    }
}