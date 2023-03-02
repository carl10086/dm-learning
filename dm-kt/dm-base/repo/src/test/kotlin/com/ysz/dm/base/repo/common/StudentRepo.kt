package com.ysz.dm.base.repo.common

import com.ysz.dm.base.core.domain.page.PageImpl
import com.ysz.dm.base.core.domain.page.PageRequest
import com.ysz.dm.base.repo.anotation.*
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
    @field:Transient
    val ignore: String? = null,
    @field: Column("create_at")
    val createAtFuck: Date = Date(),
    val age: Int? = 10,
    @field:Transient
    val gender: Int = 1,
    @field: Version
    var version: Int? = null
)


interface StudentRepo : CrudRepository<Student, Long> {
    /*jdbc*/
    fun findByUsername(username: String): Student?
    fun queryByAgeBetween(startAge: Int, endAge: Int): List<Student>
    fun queryByAgeBetweenAndUsernameOrCreateAtFuckGreaterThanEqual(
        startAge: Int, endAge: Int, username: String, startCreateAt: Date
    ): List<Student>

    fun queryByAgeInAndUsername(
        ages: List<Int>,
        username: String
    ): List<Student>

    fun searchByAgeBetween(startAge: Int, endAge: Int, page: PageRequest): PageImpl<Student>
    fun countByAgeBetween(startAge: Int, endAge: Int): Int
    fun pageTop3ByAgeBetweenOrderByIdDesc(startAge: Int, endAge: Int): List<Student>
}

object StudentSchemaTools {
    fun beforeHours(hours: Int): Date =
        Date(System.currentTimeMillis() / Duration.ofHours(1L).toMillis() - Duration.ofHours(hours.toLong()).toMillis())

    private val ds: DataSource =
        DriverManagerDataSource().apply {
            this.setDriverClassName("org.h2.Driver")
            this.url = "jdbc:h2:mem:myDb;DB_CLOSE_DELAY=-1;DB_CLOSE_ON_EXIT=FALSE;MODE=MYSQL"
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
            age int not null,
            version bigint 
            )
        """.trimIndent()
        )
        for (i in 1 until 10) {
            Student(
                username = "user$i",
                age = 10 + i,
                createAtFuck = beforeHours(i),
            ).apply { SimpleJdbcRepositoryTest.jdbcRepo.insert(this) }
        }

    }
}