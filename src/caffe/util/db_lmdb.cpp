#include "caffe/util/db_lmdb.hpp"

#include <sys/stat.h>

#include <string>

#ifdef _MSC_VER
#include <direct.h>
#include <windows.h>
#endif

namespace caffe {
	namespace db {

		const size_t LMDB_MAP_SIZE = 1099511627776;  // 1 TB

		void LMDB::Open(const string& source, Mode mode) {
			MDB_CHECK(mdb_env_create(&mdb_env_));
			MDB_CHECK(mdb_env_set_mapsize(mdb_env_, LMDB_MAP_SIZE));
			if (mode == NEW) {
#ifndef _MSC_VER
				CHECK_EQ(mkdir(source.c_str(), 0744), 0) << "mkdir " << source << "failed";
#else
				CHECK_EQ(_mkdir(source.c_str()), 0) << "mkdir " << source << "failed";
#endif
			}
			int flags = 0;
			if (mode == READ) {
				flags = MDB_RDONLY | MDB_NOTLS;
			}
			MDB_CHECK(mdb_env_open(mdb_env_, source.c_str(), flags, 0664));
			LOG(INFO) << "Opened lmdb " << source;
		}

		LMDBCursor* LMDB::NewCursor() {
			MDB_txn* mdb_txn;
			MDB_cursor* mdb_cursor;
			MDB_CHECK(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn));
			MDB_CHECK(mdb_dbi_open(mdb_txn, NULL, 0, &mdb_dbi_));
			MDB_CHECK(mdb_cursor_open(mdb_txn, mdb_dbi_, &mdb_cursor));
			return new LMDBCursor(mdb_txn, mdb_cursor);
		}

		LMDBTransaction* LMDB::NewTransaction() {
			MDB_txn* mdb_txn;
			MDB_CHECK(mdb_txn_begin(mdb_env_, NULL, 0, &mdb_txn));
			MDB_CHECK(mdb_dbi_open(mdb_txn, NULL, 0, &mdb_dbi_));
			return new LMDBTransaction(&mdb_dbi_, mdb_txn);
		}

		void LMDBTransaction::Put(const string& key, const string& value) {
			MDB_val mdb_key, mdb_value;
			mdb_key.mv_data = const_cast<char*>(key.data());
			mdb_key.mv_size = key.size();
			mdb_value.mv_data = const_cast<char*>(value.data());
			mdb_value.mv_size = value.size();
			MDB_CHECK(mdb_put(mdb_txn_, *mdb_dbi_, &mdb_key, &mdb_value, 0));
		}

		void LMDBCursor::Seek(MDB_cursor_op op) {
#ifdef _MSC_VER
			if (op != MDB_FIRST)
				VirtualUnlock(mdb_value_.mv_data, mdb_value_.mv_size);
#endif
			int mdb_status = mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, op);
			if (mdb_status == MDB_NOTFOUND) {
				valid_ = false;
			}
			else {
				MDB_CHECK(mdb_status);
				valid_ = true;
			}
		}

	}  // namespace db
}  // namespace caffe