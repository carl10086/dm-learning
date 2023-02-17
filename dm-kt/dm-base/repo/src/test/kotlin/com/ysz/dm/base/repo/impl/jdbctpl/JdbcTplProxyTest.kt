package com.ysz.dm.base.repo.impl.jdbctpl

import com.ysz.dm.base.repo.common.StudentRepo
import org.junit.jupiter.api.Test

/**
 * @author carl
 * @since 2023-02-17 4:03 PM
 */
internal class JdbcTplProxyTest {

    @Test
    fun `test bind`() {
        val studentRepo = JdbcTplProxy(
            StudentRepo::class.java
        ).getProxy()

        val interfaces = StudentRepo::class.java.genericInterfaces


        println(studentRepo.findById(1L))
    }
}