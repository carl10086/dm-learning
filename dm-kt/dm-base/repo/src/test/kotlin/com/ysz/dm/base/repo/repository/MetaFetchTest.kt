package com.ysz.dm.base.repo.repository

import com.ysz.dm.base.repo.common.Student
import com.ysz.dm.base.repo.common.StudentRepo
import com.ysz.dm.base.test.eq
import org.junit.jupiter.api.Test

/**
 * @author carl
 * @since 2023-02-17 3:40 PM
 **/
internal class MetaFetchTest {

    @Test
    fun `test fetchRepositoryType`() {
        val repositoryMetaV1 = RepositoryMetaV1(StudentRepo::class.java)
        repositoryMetaV1.idType.type eq java.lang.Long::class.java
        repositoryMetaV1.domainType.type eq Student::class.java
    }

}