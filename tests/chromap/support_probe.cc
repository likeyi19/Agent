// Guarded synthetic representation/overflow probe, linked against the exact
// isolated candidate objects. No alignment algorithm or production code here.
#include <cassert>
#include <random>
#include <cmath>
#include <tuple>
#include <limits>
#include "mapping_processor.h"
#include "mapping_writer.h"
using namespace chromap;
using Record = PairedEndMappingWithBarcode;
class Writer : public MappingWriter<Record> {
 public:
  using MappingWriter<Record>::MappingWriter;
  using MappingWriter<Record>::AppendMapping;
};
Record record(uint32_t id, uint8_t mapq, uint64_t support) {
  return Record(id, 0, 100, 100, mapq, 1, 1, support, 50, 50);
}
int main(int argc, char **argv) {
  assert(argc == 4);
  const std::string mode(argv[1]);
  if (mode == "overflow") {
    uint64_t n = std::numeric_limits<uint64_t>::max();
    IncrementDuplicateSupport(n);
    return 7;
  }
  if (mode == "summary-overflow") {
    DuplicateSupportSummaryCount(uint64_t(std::numeric_limits<int>::max()) + 1);
    return 7;
  }
  SequenceBatch ref;
  ref.InitializeLoading(argv[2]); ref.LoadAllSequences();
  MappingParameters params;
  params.mapping_output_file_path = argv[3];
  params.is_bulk_data = false;
  params.remove_pcr_duplicates = true;
  params.remove_pcr_duplicates_at_bulk_level = false;
  params.Tn5_shift = true;
  Writer writer(params, 16, {});
  if (mode == "wide") {
    for (uint64_t n : {uint64_t(255), uint64_t(256), uint64_t(65536),
         (uint64_t(1) << 32) + 7, std::numeric_limits<uint64_t>::max()}) {
      Record original = record(1, 60, n);
      Record copy = original;
      assert(copy.num_dups_ == n && copy == original);
      std::vector<std::vector<Record>> rows(1);
      rows[0].push_back(copy);
      std::vector<TempMappingFileHandle<Record>> handles;
      writer.OutputTempMappings(1, rows, handles);
      handles[0].InitializeTempMappingLoading(2);
      handles[0].LoadTempMappingBlock(1);
      assert(handles[0].GetCurrentMapping().num_dups_ == n);
      writer.AppendMapping(0, ref, handles[0].GetCurrentMapping());
      handles[0].FinalizeTempMappingLoading();
      remove(handles[0].file_path.c_str());
    }
  } else {
    MappingProcessor<Record> processor(params, 4);
    std::vector<std::vector<Record>> rows(1);
    std::vector<TempMappingFileHandle<Record>> handles;
    for (int i = 0; i < 300; ++i) {
      rows[0].push_back(record(i, i == 299 ? 60 : 10, 1));
      if (mode == "spill" && (i == 149 || i == 299)) {
        processor.SortOutputMappings(1, rows);
        writer.OutputTempMappings(1, rows, handles);
      }
    }
    if (mode == "spill") {
      writer.ProcessAndOutputMappingsInLowMemory(0, 1, ref, nullptr, handles);
    } else {
      assert(mode == "memory");
      processor.RemovePCRDuplicate(1, rows, 1);
      assert(rows[0].size() == 1 && rows[0][0].num_dups_ == 300);
      assert(rows[0][0].mapq_ == 60 && rows[0][0].read_id_ == 299);
      processor.ApplyTn5ShiftOnMappings(1, rows);
      writer.OutputMappings(1, ref, rows);
    }
  }
  ref.FinalizeLoading();
}
