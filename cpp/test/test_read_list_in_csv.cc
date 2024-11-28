#include <arrow/api.h>
#include <arrow/csv/api.h>
#include <arrow/io/api.h>
#include <arrow/table.h>
#include <arrow/type.h>
#include <iostream>
#include <sstream>

std::shared_ptr<arrow::Table> read_csv_to_table(const std::string& filename) {
    arrow::csv::ReadOptions read_options;
    arrow::csv::ParseOptions parse_options;
    arrow::csv::ConvertOptions convert_options;
    
    parse_options.delimiter = '|';

    auto input_result = arrow::io::ReadableFile::Open(filename, arrow::default_memory_pool());
    if (!input_result.ok()) {
        std::cerr << "Error opening file: " << input_result.status().message() << std::endl;
        return nullptr;
    }

    auto reader_result = arrow::csv::TableReader::Make(arrow::io::default_io_context(),
                                                       input_result.ValueOrDie(),
                                                       read_options,
                                                       parse_options,
                                                       convert_options);
    if (!reader_result.ok()) {
        std::cerr << "Error creating CSV reader: " << reader_result.status().message() << std::endl;
        return nullptr;
    }

    auto table_result = reader_result.ValueOrDie()->Read();
    if (!table_result.ok()) {
        std::cerr << "Error reading table: " << table_result.status().message() << std::endl;
        return nullptr;
    }

    return table_result.ValueOrDie();
}

int main() {
    auto table = read_csv_to_table("test.csv");
    if (!table) {
        return -1;
    }

    std::cout << "Original Table:\n" << table->ToString() << std::endl;

    auto values_array = table->GetColumnByName("Synoniem")->chunk(0);
    auto values_column = std::dynamic_pointer_cast<arrow::StringArray>(values_array);
    if (!values_column) {
        std::cerr << "Error: column 'Synoniem' is not a string column." << std::endl;
        return -1;
    }

    arrow::MemoryPool* pool = arrow::default_memory_pool();
    auto value_builder = std::make_shared<arrow::StringBuilder>(pool);  // Use a pointer for managing builder
    arrow::ListBuilder list_builder(pool, value_builder);

    for (int64_t i = 0; i < values_column->length(); ++i) {
        if (values_column->IsValid(i)) {
            std::string raw_values = values_column->GetString(i);
            std::istringstream ss(raw_values);
            std::string item;

            list_builder.Append();  
            auto inner_builder = static_cast<arrow::StringBuilder*>(list_builder.value_builder());
            while (std::getline(ss, item, ';')) {
                inner_builder->Append(item);
            }
        } else {
            list_builder.AppendNull();
        }
    }
    
    std::shared_ptr<arrow::Array> list_array;
    auto status = list_builder.Finish(&list_array);
    if (!status.ok()) {
        std::cerr << "Error finishing list builder: " << status.message() << std::endl;
        return -1;
    }

    std::cout << "Converted List Array:\n" << list_array->ToString() << std::endl;

    return 0;
}
